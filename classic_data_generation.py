import gym
import time

import numpy as np

from spinup.utils.test_policy import load_policy_and_env
from expert_object import ExpertObject

if __name__ == "__main__":
    env_names = ["MountainCarContinuous-v0", "MountainCar-v0", "CartPole-v1", "Acrobot-v1", "Pendulum-v0"]
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
                log_path = "log_classic/classic_%s_%s_%d/classic_%s_%s_%d_s%d" \
                        %(algo, env_id, seed, algo, env_id, seed, seed)

                _, pi_fn = load_policy_and_env(log_path)
                for i in range(rounds):
                    ex_object.collect_one_traj(pi_fn, save_to_file="classic_%s-%d-%d"%(algo, seed, i))
                    time.sleep(1.0)
