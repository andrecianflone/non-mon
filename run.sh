#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED} $1${CLEAR}\n";
  fi
  exit 1
}

# Debug or not
BASE='python main.py'
if [ -n "$2" ]; then
    if [[ $2 == *"debug"* ]]; then
        BASE='python -m pudb main.py'
    fi
fi

case "$1" in
        1)
            echo "Baseline ON 1 layer"
            eval $BASE \
                --nlayers 1 \
                --batch_size 20 \
                --dropout 0.45 \
                --dropouth 0.3 \
                --dropouti 0.5 \
                --wdrop 0.45 \
                --chunk_size 10 \
                --seed 141 \
                --epoch 1000
            ;;

        2)
            echo "Non-mon"
            eval $BASE \
                --nlayers 1 \
                --model "NMLSTM" \
                --batch_size 20 \
                --dropout 0.45 \
                --dropouth 0.3 \
                --dropouti 0.5 \
                --wdrop 0.45 \
                --chunk_size 10 \
                --seed 141 \
                --epoch 1000
            ;;

        3)
            echo "Baseline ON"
            eval $BASE \
                --batch_size 20 \
                --dropout 0.45 \
                --dropouth 0.3 \
                --dropouti 0.5 \
                --wdrop 0.45 \
                --chunk_size 10 \
                --seed 141 \
                --epoch 1000
            ;;

        *)
            usage "You need to call $0 with an int option"
            exit 1
esac

