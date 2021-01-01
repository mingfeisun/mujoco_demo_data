import numpy as np
import matplotlib.pyplot as plt
from spinup.utils.plot import get_all_datasets,  plot_data

if __name__ == "__main__":
    legend = None
    select = None
    exclude = None
    values  = 'Performance'
    estimator = 'mean'
    xaxis = 'TotalEnvInteracts'
    count = True
    smooth = 1

    env_names = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Walker2d-v3", "Swimmer-v3"]
    algos = ["PPO", "DDPG", "SAC", "TD3"]
    seeds = [0, 1, 2, 3, 4]

    for env_id in env_names:
        env_id = env_id.lower().replace('-', '_')
        for algo in algos:
            algo = algo.lower()
            for seed in seeds:
                log_path = "log/mujoco_%s_%s_%d" \
                        %(algo, env_id, seed)

                data = get_all_datasets([log_path], legend, select, exclude)
                values = values if isinstance(values, list) else [values]
                condition = 'Condition2' if count else 'Condition1'
                attr_estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
                for value in values:
                    plt.figure()
                    plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=attr_estimator)
                    plt.savefig('%s/%s.png'%(log_path, value))