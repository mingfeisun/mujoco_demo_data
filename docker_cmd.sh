#!/usr/bin/env bash

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU=$1 ${cmd} run --rm --user $(id -u) \
                    -v `pwd`:/home/mingfei/data \
                    --entrypoint '/bin/bash' imitation_spinningup:1.0 \
                    -c "OMP_NUM_THREADS=1 $2"
