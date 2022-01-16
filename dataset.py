import copy
import random
import numpy as np

from pathlib import Path

class Dataset:
    def __init__(self, env_id, max_num_trajs=10000, dir_samples=None):
        self.idx = 0
        self.max_num_trajs = max_num_trajs
        self.env_id = env_id
        self.picked_trajs = None
        self.trajs = []
        self.traj_rets = []
        self.traj_tags = []
        self.meta_info = []
        self.transitions = []
        self.transitions_sampled = []
        self.transition_tags = []
        self.oracle_fn = None
        self.dir_samples = None
        self.traj_filename = None
        if dir_samples != None:
            self.dir_samples = dir_samples
            Path(self.dir_samples).mkdir(parents=True, exist_ok=True)

    @property
    def num_trajs(self):
        return len(self.trajs)

    @property
    def num_transitions(self):
        return len(self.transitions)

    def get_trajs(self):
        return copy.deepcopy(self.trajs)

    def set_oracle(self, oracle_fn):
        self.oracle_fn = oracle_fn

    def set_trajs(self, demo_trajs, demo_traj_rets, traj_tags):
        if len(demo_trajs) > self.max_num_trajs:
            self.trajs = copy.deepcopy(demo_trajs[:self.max_num_trajs])
            self.traj_rets = copy.deepcopy(demo_traj_rets[:self.max_num_trajs])
            self.traj_tags = copy.deepcopy(traj_tags[:self.max_num_trajs])
        else:
            self.trajs = copy.deepcopy(demo_trajs)
            self.traj_rets = copy.deepcopy(demo_traj_rets)
            self.traj_tags = copy.deepcopy(traj_tags)
        self.make_transitions()

    def get_transitions(self):
        return copy.deepcopy(self.transitions)

    def get_sasa_transitions(self):
        return copy.deepcopy(self.transitions)

    def clear(self):
        self.trajs = []
        self.traj_rets = []
        self.traj_tags = []
        self.transitions = []

    def feed(self, states, actions, next_states, rewards, dones, infos):
        traj = copy.deepcopy(list(zip(states, actions, next_states, rewards, dones, infos)))
        traj_ret = sum(rewards)
        self.trajs.append(traj)
        self.traj_rets.append(traj_ret)
        self.transitions += traj

    def feed_one_transition(self, s, a, n_s, r, d, info):
        tran = copy.deepcopy((s, a, n_s, r, d, info))
        self.transitions_sampled.append(tran)
        if len(self.transitions_sampled) > len(self.transitions)//4:
            self.transitions_sampled.pop(0)
    
    def load(self, filename):
        if self.dir_samples:
            filename = "%s/%s"%(self.dir_samples, filename)
        data = np.load(filename)
        self.trajs = data['trajs']
        self.traj_rets = data['traj_rets']
        self.traj_tags = data['traj_tags']
        self.make_transitions()

    def load_traj_from_files(self, filename, expert_data_folder='rnn_traj', repeat_trajs=False):
        self.traj_filename = filename.split("/")[-1]
        print('Loading expert data...')
        with open(filename) as fin:
            line = fin.readline().strip() # remove \n
            while line:
                if self.env_id not in line:
                    line = fin.readline().strip() # remove \n
                    continue
                traj_file = "%s/%s"%(expert_data_folder, line)
                data = np.load(traj_file, allow_pickle=True)
                trajs, traj_rets, meta = data['trajs'], data['traj_rets'], data['meta']
                meta = meta.flatten()[0]
                meta_env_id, meta_info = meta['env_id'], meta['info']

                self.trajs.append(trajs.tolist())
                self.traj_tags.append(meta_info)
                traj_rets = traj_rets.flatten()[0]
                self.traj_rets.append(traj_rets)
                self.meta_info.append(meta_info)

                line = fin.readline().strip() # remove \n

        # self.plot_return_hist()
        if repeat_trajs:
            self.make_transitions_with_repeated_trajs()
        else:
            self.make_transitions()
        print('Expert data loaded.')

    def save(self, _filename):
        if self.dir_samples:
            filename = "%s/%s-%s_len_%d_return_%.2f"%(self.dir_samples, _filename, self.env_id, len(self.trajs), self.traj_rets)
        meta = {"env_id": self.env_id, "info": _filename}
        np.savez(filename, trajs=self.trajs, traj_rets=self.traj_rets, meta=meta)

    def make_transitions(self):
        if self.picked_trajs:
            trajs = [self.trajs[i] for i in self.picked_trajs]
            tags = [self.traj_tags[i] for i in self.pick_trajs]
        else:
            trajs = self.trajs
            tags = self.traj_tags

        self.transitions = []
        self.transition_tags = []

        # traj: (states, actions, next_states, rewards, dones, infos)
        # ATTENTION: handling done signal here
        max_traj_length = 1000 # the default value is 1000 in Mujoco domains
        for idx, traj in enumerate(trajs):
            states, actions, next_states, rewards, dones, infos = zip(*traj)
            # trajectory is maximum
            if len(traj) == max_traj_length:
                dones = list(dones)
                dones[-1] = False
                dones = tuple(dones)
            self.transitions += list(zip(states, actions, next_states, rewards, dones, infos))
            try: 
                self.transition_tags += [tags[idx]] * len(traj)
            except:
                pass

    def pick_trajs(self, picked_trajs):
        assert np.max(picked_trajs) < self.num_trajs
        self.picked_trajs = picked_trajs
        self.make_transitions()

    def sample_trajs(self, batch_size=1):
        return [self.trajs[random.randint(0, self.num_trajs)] for _ in range(batch_size)]

    def gen_sample_indices(self, batch_size, replace, mode='sa'):
        if mode == 'sa':
            num_transitions = len(self.transitions)
        else:
            raise NotImplementedError

        if batch_size < 0:
            return np.arange(num_transitions)

        if replace:
            sample_indices = [np.random.randint(0, num_transitions) for _ in range(batch_size)]
        else:
            sample_indices = np.random.choice(num_transitions, batch_size, replace=False)
        return sample_indices

    def gen_sample_indices_mixed(self, batch_size, replace):
        num_transitions = len(self.transitions + self.transitions_sampled)
        if batch_size < 0:
            return np.arange(num_transitions)
        if replace:
            sample_indices = [np.random.randint(0, num_transitions) for _ in range(batch_size)]
        else:
            sample_indices = np.random.choice(num_transitions, batch_size, replace=False)
        return sample_indices

    def get_next_batch(self, batch_size):
        return self.sample(batch_size)

    def get_all_tuples(self):
        samples = list(zip(*self.transitions))
        return samples

    def get_all_tuples_next_action(self):
        samples = list(zip(*self.transitions))
        return samples

    def sample(self, batch_size=128, replace=True):
        samples = [self.transitions[i] \
                        for i in self.gen_sample_indices(batch_size, replace)]
        samples = list(zip(*samples))
        return samples

    def sample_mixed(self, batch_size=128, replace=True):
        tmp_transitions = self.transitions + self.transitions_sampled
        samples = [tmp_transitions[i] \
                        for i in self.gen_sample_indices_mixed(batch_size, replace)]
        samples = list(zip(*samples))
        return samples

    def sample_s(self, batch_size=128, replace=True):
        samples = [self.transitions[i][0] \
                        for i in self.gen_sample_indices(batch_size, replace)]
        return samples

    def sample_sa(self, batch_size=128, replace=True):
        samples = [(self.transitions[i][0], self.transitions[i][1]) \
                        for i in self.gen_sample_indices(batch_size, replace)]
        samples = list(zip(*samples))
        return samples
    
    def sample_sasr(self, batch_size=128, replace=True):
        samples = [(self.transitions[i][0], self.transitions[i][1], self.transitions[i][2], self.transitions[i][3]) \
                        for i in self.gen_sample_indices(batch_size, replace, mode='sasr')]
        samples = list(zip(*samples))
        return samples

    def plot_return_hist(self):
        def mkdir(path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
        
        import pandas as pd
        import seaborn as sns
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        data = {'returns': self.traj_rets}
        data = pd.DataFrame(data)
        plot = sns.histplot(data, x='returns', kde=True)

        mkdir('figures')
        fig = plot.get_figure()
        if self.traj_filename is None:
            self.traj_filename = self.env_id
        fig.savefig("figures/demonstration_rets_%s.png"%(self.traj_filename))

    def visualize_distribute(self):
        import seaborn as sns
        sns.set(rc={'figure.figsize':(11.7,8.27)})

        states, actions, tags = [], [], []
        for i in range(len(self.transitions)):
            states.append(self.transitions[i][0])
            actions.append(self.transitions[i][1])
            # extract the policy tag
            tags.append(self.transition_tags[i].split('-')[0]) 

        X = np.concatenate((states, actions), axis=-1)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='random', random_state=0, n_iter=1000, perplexity=30.0)
        print('Fitting to data...')
        X_embedded = tsne.fit_transform(X)
        print('Done fitting.')

        import pandas as pd 
        data = {"X": X_embedded[:, 0], "Y": X_embedded[:, 1], "policy": tags}
        data = pd.DataFrame(data)
        plot = sns.scatterplot(data=data, x="X", y="Y", hue="policy", style="policy")

        fig = plot.get_figure()
        if self.traj_filename is None:
            self.traj_filename = self.env_id
        fig.savefig("figures/sa_distribution_%s.png"%(self.traj_filename))

if __name__ == "__main__":
    test = Dataset(env_id='Ant-v2')
    expert_trajs_file = 'traj_configs/Ant.txt'
    test.load_traj_from_files(expert_trajs_file)

    print(test.sample_sa())