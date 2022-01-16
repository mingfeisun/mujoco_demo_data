import time
from stable_baselines import PPO2
from expert_object import ExpertObject

def load_policy_fn(env_id):
    load_dir = 'log/PPO2.RNN.{}/model.zip'.format(env_id)
    model = PPO2.load(load_dir)
    return model.predict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Hopper-v2')
    args = parser.parse_args()

    env_names = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2"]
    rounds = 10
    dir_samples = "rnn_data"
    import time

    for env_id in env_names:
        ex_object = ExpertObject(env_id, dir_samples=dir_samples)
        vec_pi_fn = load_policy_fn(env_id)
        for i in range(rounds):
            ex_object.collect_one_traj(vec_pi_fn, save_to_file="PPO-RNN-%03d"%(i))
            time.sleep(1.0)
