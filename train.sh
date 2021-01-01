cat tasks.txt | xargs -n 4 -P 30 \
    sh -c 'bash docker_cmd.sh 0 "python3 -m spinup.run $0 --env $1 --seed $2 --exp_name $3 --data_dir log --epochs 500 --steps_per_epoch 4000"'
