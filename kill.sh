#!/usr/bin/env bash
if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

# kill & remove container
${cmd} ps -a | awk '{print $1,$2}' | grep imitation_spinningup:1.0 | awk '{print $1}' | xargs -iz ${cmd} kill z
${cmd} ps -a | awk '{print $1,$2}' | grep imitation_spinningup:1.0 | awk '{print $1}' | xargs -iz ${cmd} rm z
