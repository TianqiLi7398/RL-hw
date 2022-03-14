""" 
This document is the DQN practice hw for ECEN 689 RL
Author: Tianqi Li
Date: Mar 13, 2022
The problem is lunarlander-v2: 
https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py



 """


import numpy as np
from utils.learner import Q_NN, memoryBuffer
from utils.agent import Agent
import torch
import gym
from collections import deque

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
TARGET_UPDATE = 4        # how often to update the network
EPS_START = 0.9         # epsilon-greedy parameter with decay
EPS_END = 0.05
EPS_DECAY = 200


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iteration_num = 2e4
    t_max = 1e4
    dqn(device, iteration_num, t_max)
    
    
def dqn(device, iteration_num, t_max, eps_start = EPS_START, eps_end = EPS_END,
        eps_decay = EPS_DECAY, sliding_size = 100):
    

    # initialize agent (memory, network inside agent class)

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = Agent(env.observation_space.shape, env.action_space, seed=0, device=device)

    scores_record, scores_sliding_record = [], []
    scores_window = deque(max_len = sliding_size)
    eps = eps_start
    for epsiode in range(iteration_num):
        state = env.reset()
        score  = 0
        for t in range(t_max):
            action = agent.select_action(state, eps = eps)
            next_state, reward, terminal, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminal)
            state = next_state
            score += reward
            if terminal:   break
        
        # update the frozen network, copying all weights and bies in DQN
        if epsiode % TARGET_UPDATE == 0:
            agent.update_frozen_nn()

        # record scores and sliding scores
        scores_record.append(score)
        if len(scores_window) == sliding_size:
            # calculate sliding window average
            score_to_pop = scores_window[0]
            try:
                previous_average = scores_sliding_record[-1]
                new_average = previous_average + (score - score_to_pop) / sliding_size
            except:
                new_average = np.mean(scores_window)
            scores_sliding_record.append(new_average)
        scores_window.append(score)
        # decay the eps for more exploitation
        eps = max(eps_end, eps*eps_decay)

        # critia to stop training, TODO
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    # save the results here, TODO
    
    return
        



if __name__ == "__main__":
    main()