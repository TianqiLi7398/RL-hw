""" 
This document is the DQN practice hw for ECEN 689 RL
Author: Tianqi Li
Date: Mar 13, 2022
The problem is lunarlander-v2: 
https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
Please check the README.md for more info of using this.


 """


from glob import escape
from random import uniform
import numpy as np
# from utils.learner import Q_NN, memoryBuffer
from utils.agent import Agent
from utils.util import util
import torch
import gym
from gym.wrappers.monitoring import video_recorder
from collections import deque
import os, sys
import json
import time

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
TARGET_UPDATE = 4       # how often to update the network
EPS_START = 0.9         # epsilon-greedy parameter with decay
EPS_END = 0.05
EPS_DECAY = 0.995


def main(args, task = 'train', dqn_type = 'dqn_with_memoryreplay', eposide = int(5e3)):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(args) > 1:
        try: 
            task, dqn_type, episode = args[1], args[2], args[3]
            
        except:
            print("pring run 'python3 task main.py dqn_type epsiode' as command to plot scores")
            return
        
        if task == 'plot':
            analysis_scores(dqn_type, episode)
            return
        elif task == 'video':
            animate_generate(dqn_type, episode, device)
            return
    
    iteration_num = int(episode)
    t_max = int(1e4)
    print("start training with %s, t_max %s, iteration number %s" % (dqn_type, t_max, iteration_num))
    dqn(device, iteration_num, t_max, train_type=dqn_type)
    
    
def dqn(device, iteration_num, t_max, eps_start = EPS_START, eps_end = EPS_END,
        eps_decay = EPS_DECAY, sliding_size = 100, train_type = 'dqn_with_memoryreplay'):
    

    # initialize agent (memory, network inside agent class)
    story = util.name_file(train_type)

    if train_type == 'dqn_with_memoryreplay':
        isReplay, uniform = True, False
        print("i have memory replay")
    elif train_type == 'dqn_without_memoryreplay':
        isReplay, uniform = False, False
        print("i don't have memory replay")
    elif train_type == 'dqn_uniform_behavior':
        isReplay, uniform = True, True
    else:
        raise RuntimeError("please specify the correct training method")
        
    start_time = time.time()

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = Agent(8, 4, seed=0, device=device,
            batch_size = BATCH_SIZE,
            buffer_size = BUFFER_SIZE,
            isReplay= isReplay, uniform = uniform)

    scores_record, scores_sliding_record = [], []
    scores_window = deque(maxlen = sliding_size)
    eps = eps_start
    for episode in range(iteration_num):
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
        if episode % TARGET_UPDATE == 0:
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
        eps = max(eps_end, eps * eps_decay)

        # save checkpoints
        if episode % 500 == 0 and episode > 0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, scores_sliding_record[-1]))
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            save_scores(episode, scores_record, scores_sliding_record, story, sliding_size)
            agent.save_checkpoint(episode, story)
            # break
            print("time used till now %s" % (time.time() - start_time))

    
    return

def save_scores(t, scores_record, scores_sliding_record, story, sliding_size):
    filename = os.path.join(os.getcwd(), 'model', '%s_scores_%s.json' % (story, t))
    data = {
        'sliding_size': sliding_size,
        'scores': scores_record,
        'scores_sliding': scores_sliding_record
    }

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)
        
def analysis_scores(dqn_type, episode):

    util.plot_sliding(episode, dqn_type)

def animate_generate(dqn_type, episode, device):
    env = gym.make('LunarLander-v2')
    filename = os.path.join(os.getcwd(), 'videos', '%s%s.mp4'% (util.name_file(dqn_type), episode))
    vid = video_recorder.VideoRecorder(env, path= filename)

    # init an agent
    agent = Agent(8, 4, seed=0, device=device,
            batch_size = BATCH_SIZE,
            buffer_size = BUFFER_SIZE)

    model_file = os.path.join(os.getcwd(), 'model', '%s_nn_model_%s.pth' % (util.name_file(dqn_type), episode))
    agent.q_nn_update.load_state_dict(torch.load(model_file))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        
        action = agent.select_action(state)

        state, reward, done, _ = env.step(action)        
    env.close()

if __name__ == "__main__":
    # sys.argv[]
    main(sys.argv)