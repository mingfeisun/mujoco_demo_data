#!/usr/bin/env bash
nvidia-docker run --rm --user $(id -u) -v `pwd`:/home/$USER/data -it imitation_rnn:1.0