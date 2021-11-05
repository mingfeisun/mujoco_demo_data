import gym
import time

import numpy as np

from spinup.utils.test_policy import load_policy_and_env
from expert_object import ExpertObject

if __name__ == "__main__":
    # env_names = ["BipedalWalker-v2", "LunarLanderContinuous-v2"]
    env_names = ["LunarLanderContinuous-v2"]
    algos = ["PPO"]
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
                log_path = "log_box2d/%s_%s_%d/%s_%s_%d_s%d" \
                        %(algo, env_id, seed, algo, env_id, seed, 0)

                _, pi_fn = load_policy_and_env(log_path)
                for i in range(rounds):
                    ex_object.collect_one_traj(pi_fn, save_to_file="box2d_%s-%d-%d"%(algo, seed, i))
                    time.sleep(1.0)
