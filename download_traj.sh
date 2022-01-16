# download and unzip
wget https://github.com/mingfeisun/mujoco_demo_data/archive/refs/heads/rnn_traj.zip -O data.zip \
  && unzip data.zip -d tmp_data \
  && rm data.zip

# move data
mv tmp_data/mujoco_demo_data-rnn_traj/rnn_data rnn_traj
rm -rf tmp_data