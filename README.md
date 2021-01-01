# mujoco_demo_data

This repo generates demonstration data for Mujoco domains with reinforcement learning algorithms. 
*Mujoco environments*: Ant-v3, HalfCheetah-v3, Hopper-v3, Walker2d-v3, Swimmer-v3. 
*RL algorithms*: PPO, TD3, SAC, DDPG.

## How to use
* Build docker image first
``` bash
bash docker_build.sh # this builds an image named imitation_spinningup
```

* Running training and generating logs
``` bash 
bash train.sh # this runs trainings and logs results in log folder
```

* Generating demonstrations
``` bash
python3 mujoco_data_generation.py # this produces demonstrations in folder expert_data
```
