#!/usr/bin/env bash
NV_GPU=$1 nvidia-docker run --rm --user $(id -u) \
                    -v `pwd`:/home/$USER/data \
                    --entrypoint '/bin/bash' imitation_rnn:1.0 \
                    -c "OMP_NUM_THREADS=1 $2"