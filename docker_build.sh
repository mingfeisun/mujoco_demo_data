#!/usr/bin/env bash
nohup docker build --build-arg UID=$UID -t imitation_spinningup:1.0 . &
