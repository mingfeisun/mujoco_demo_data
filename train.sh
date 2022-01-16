cat tasks.txt | xargs -n 3 -P 50 \
    sh -c 'bash docker_cmd.sh $0 "python3 rnn_training.py --env_id=$1 --total_timesteps=$2"'
