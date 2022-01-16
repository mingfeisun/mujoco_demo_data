## How to use
* First download the dataset
```bash
bash download_traj.sh 
```

* Second specify the trajectories as training/testing data
```bash
# go to traj_configs
# examples are given for ant/halfcheetah/hopper/swimmer/walker2d
Ant_train.txt # training on trajectories generated via PPO-RNN
Ant_test.txt # testing on trajectories generated via PPO-RNN
```

* Training/testing with `dataset.py`
```python3
from dataset import Dataset

# specify the Mujoco domains
trainset = Dataset(env_id='Ant-v2')
testset = Dataset(env_id='Ant-v2')

# specify the training/testing trajectory files
trajs_file_train = 'traj_configs/Ant_train.txt'
trajs_file_test = 'traj_configs/Ant_test.txt'

# loading data
trainset.load_traj_from_files(trajs_file_train)
testset.load_traj_from_files(trajs_file_test)

# sampling; return (s, a) tuples
train_s, train_a = trainset.sample_sa()
test_s, test_a = testset.sample_sa()
```