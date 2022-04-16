import torch
import numpy as np

device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(0)

action_dim = 2
state_dim = 2

dtype = torch.float64
torch.set_default_dtype(dtype)
np.random.seed(0)
torch.manual_seed(0)
theta = torch.normal(0, 0.01, size=(action_dim, state_dim + 1))
policy_net = None
theta = theta.to(dtype).to(device)
print("init, %s" % theta.get_device())
# change to cpy

theta.to('cpu').numpy()
print("after change, %s" % theta.get_device())

#!/usr/bin/env python

import gym
import gym.wrappers

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.metadata["render.modes"] = ["human", "rgb_array"]
env = gym.wrappers.Monitor(env=env, directory="./videos", force=True)

episodes = 10
_ = env.reset()

done = False
while episodes > 0:
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        _ = env.reset()
        episodes -= 1
