from pathlib import Path
from stable_baselines import PPO2

def get_save_dir(args):
    Path('log').mkdir(parents=True, exist_ok=True)
    save_dir = 'log/PPO2.RNN.{}/'.format(args.env_id)
    return save_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Hopper-v2')
    parser.add_argument('--nminibatches', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=int(2e4))
    args = parser.parse_args()

    # For recurrent policies, with PPO2, the number of environments run in parallel
    # should be a multiple of nminibatches.
    save_dir = get_save_dir(args)
    
    # number of parallel envs: 1
    model = PPO2('MlpLstmPolicy', args.env_id, 
                nminibatches=args.nminibatches, 
                verbose=args.verbose, 
                tensorboard_log=save_dir)

    model.learn(args.total_timesteps)
    model.save('%s/model'%save_dir)