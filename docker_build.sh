#!/usr/bin/env bash
nohup docker build --build-arg UID=$UID --build-arg USER=$USER -t imitation_rnn:1.0 . &