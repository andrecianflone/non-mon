import torch.nn.functional as F
import torch.nn as nn
import torch

from locked_dropout import LockedDropout

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
                # dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class nmONLSTMCell(nn.Module):
    """ Where the ON calc happens """

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]

        # Future state params
        self.weight_hgs, self.weight_gsh, self.weight_hhf = \
                                                    self.get_future_state()

    def get_future_state_params(self):
        # Future state params
        # Weights to produce the GS scores
        weight_hgs = nn.Parameter(torch.FloatTensor(hidden_size, window))
        # Weights to interact GS sample and hidden
        weight_gsh = nn.Parameter(torch.FloatTensor(hidden_size + window,
                                                                hidden_size))
        # Weights to predict future hidden
        weight_hhf = nn.Parameter(torch.FloatTensor(hidden_size,
                                                                hidden_size))

        # Initialize parameters
        # TODO: better init and bias

        # Uniform init for now
        initrange = 0.1
        weight_hgs.data.uniform_(-initrange, initrange)
        weight_gsh.data.uniform_(-initrange, initrange)
        weight_hhf.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input) # input to hidden product

        # Eq. 1 to 4, compute all hiddens and add inputs
        gates = transformed_input + self.hh(hx)
        # `cingate` and `cforgetgate` are the tilde-in and tilde-f gates
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:].view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        # Eq. 9 and 10
        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        # TODO: What are the distances for?
        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate # Eq. 11
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell # Eq. 14

        # hy = outgate * torch.tanh(self.c_norm(cy))
        hy = outgate * torch.tanh(cy)

        # Get non-mon tensors
        h_f, h_c, loc = self.nm_forward(hy,

        return hy.view(-1, self.hidden_size), cy,
                            (distance_cforget, distance_cin), (h_f, h_c, loc)

    def nm_forward(self, h_1):
        """
        Non-mon forward part of the cell
        Arguments:
            h_1: current hidden state
        """
        # Predict future hidden
        gs_logprob = torch.mm(h_1, self.weight_hgs)

        # Future position: greedy or sampled
        if not self.training and self.greedy_eval:
            onehot = torch.argmax(gs_logprob, dim=1)
            sample = torch.nn.functional.one_hot(onehot,
                                                num_classes=gs_logprob.size(1))
            sample = sample.type_as(gs_logprob)
        else:
            # TODO: Add dirichlet prior? Concentrate near current timestep?
            bern = LogitRelaxedBernoulli(1, logits=gs_logprob)
            sample = bern.rsample()

        # Get the hard location to decide on the target for future hidden
        loc = torch.argmax(sample, dim=1)

        # Concat gs_sample at end of hidden
        c = torch.cat((h_1, sample), dim=1)

        # Interact 1
        h_f = torch.relu(torch.mm(c, self.weight_gsh))

        # Interact 2
        h_f = torch.relu(torch.mm(h_f, self.weight_hhf))

        # Past and future are combined
        # TODO: Additive?
        h_c = h_1 + h_f

        return h_f, h_c, loc


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, dropout=0., dropconnect=0.):
        super(ONLSTMStack, self).__init__()
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, hidden):
        length, batch_size, _ = input.size()

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = []
        outputs = []
        distances_forget = []
        distances_in = []

        # Non-mon lists storing each layer...
        h_n_c = [] # lists of output composed of past and future hidden
        h_n_f = [] # lists of future hidden
        loc = [] # lists of future locations

        # Loop RNN Layers
        for l in range(len(self.cells)):
            curr_layer = [None] * length # to store time step hidden
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            # Non-mon lists for the sequence
            h_future = [] # future hidden states
            output_c = [] # output hidden, combined h and h_future
            loc_future = [] # int location of future, increment by time?

            # Loop over the sequence inputs
            for t in range(length):
                # `d` is distance vectors, `nm` non-mon vectors
                hidden, cell, d, nm = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                # Non-mon states
                hf, h_c, loc = nm

                # Ordered neurons params
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

                # Non-mon stuff
                # `loc` is relative location, make absolute
                loc += t + 1 # + 1 -> don't allow to predict self
                # mask = (t < length).float().unsqueeze(1).expand_as(hidden)
                # h_next = h_next*mask + hx[0]*(1 - mask)
                # c_next = c_next*mask + hx[1]*(1 - mask)
                # hx_next = (h_next, c_next)
                output_c.append(h_c)
                h_future.append(hf)
                loc_future.append(loc)

            # Stack sequence lists
            prev_layer = torch.stack(curr_layer)
            dist_cforget, dist_cin = zip(*dist)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs.append(prev_layer)
            distances_forget.append(dist_layer_cforget)
            distances_in.append(dist_layer_cin)

            # Non-mon states stack
            # TODO: Should have this for each layer?
            layer_out_c = torch.stack(output_c, 0)
            layer_h_future = torch.stack(h_future, 0)
            layer_loc_future = torch.stack(loc_future, 0)

            h_n_c.append(layer_out_c)
            h_n_f.append(layer_h_future)
            loc.append(layer_loc_future)

        output = prev_layer

        # Non-mon targets
        output = layer_out_c # combined past-future
        h_past = torch.stack(h_past, 0)
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        h_n_c = torch.stack(h_n_c, 0)
        h_n_f = torch.stack(h_n_f, 0)
        loc = torch.stack(loc, 0)

        # Cap location array
        loc[loc>=length]=0

        # Swap dims so it makes sense
        loc = loc.permute(0, 2, 1)
        h_past = h_past.permute(0, 2, 1, 3)
        # Gather h_past indexed by location
        loc = loc.unsqueeze(3).expand(h_past.shape)
        trg = h_past.gather(2, loc)
        # Set trg to 0 if beyond max_time
        mask = loc>0
        mask.requires_grad = False
        trg = trg*mask.type_as(trg)
        trg = trg.permute(0, 2, 1, 3)

        return output, prev_state, raw_outputs, outputs, (torch.stack(distances_forget), torch.stack(distances_in)), (h_n_f, trg)


if __name__ == "__main__":
    x = torch.Tensor(10, 10, 10)
    x.data.normal_()
    lstm = ONLSTMStack([10, 10, 10], chunk_size=10)
    print(lstm(x, lstm.init_hidden(10))[1])

