#!/usr/bin/env bash
# kill & remove container
nvidia-docker ps -a | awk '{print $1,$2}' | grep imitation_rnn:1.0 | awk '{print $1}' | xargs -iz ${cmd} kill z
nvidia-docker ps -a | awk '{print $1,$2}' | grep imitation_rnn:1.0 | awk '{print $1}' | xargs -iz ${cmd} rm z