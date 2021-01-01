import gym
import time

import numpy as np

from spinup.utils.test_policy import load_policy_and_env
from expert_object import ExpertObject

if __name__ == "__main__":
    # env_names = ["Ant-v3", "Hopper-v3", "HalfCheetah-v3", "Walker2d-v3", "Swimmer-v3"]
    # algos = ["PPO", "TRPO", "SAC", "TD3", "DDPG"]
    # env_names = ["HalfCheetah-v3"]
    env_names = ["Ant-v3", "Hopper-v3", "Walker2d-v3", "Swimmer-v3"]
    algos = ["PPO", "SAC", "TD3", "DDPG"]
    seeds = [0, 1, 2, 3, 4]

    rounds = 10
    dir_samples = "expert_data"
    import time

    for env_id in env_names:
        ex_object = ExpertObject(env_id, dir_samples=dir_samples)
        env_id = env_id.lower().replace('-', '_')
        for algo in algos:
            algo = algo.lower()
            for seed in seeds:
                log_path = "log/mujoco_%s_%s_%d/mujoco_%s_%s_%d_s%d" \
                        %(algo, env_id, seed, algo, env_id, seed, seed)

                _, pi_fn = load_policy_and_env(log_path)
                for i in range(rounds):
                    ex_object.collect_one_traj(pi_fn, save_to_file="%s-%d-%d"%(algo, seed, i))
                    time.sleep(1.0)
