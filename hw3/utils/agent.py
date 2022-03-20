""" 
This document is the DQN practice hw for ECEN 689 RL
Author: Tianqi Li
Date: Mar 13, 2022
1 class is defined

Agent: maintains the Q-approximate class Q_NN and memory buffer, 
    selects actions based on Q value

 """
import random
import numpy as np
from utils.learner import Q_NN, memoryBuffer
import torch
import torch.optim as optim
import torch.nn.functional as F
import os, json

class Agent:

    def __init__(self, state_space, action_space, 
            seed = 0,
            device = 'cpu', 
            learn_rate = 5e-4,
            gamma = 0.99, 
            batch_size = 10,
            buffer_size = int(1e5),
            isReplay = True, 
            uniform = False):
        
        # parameter definition
        self.state_space = state_space
        self.action_space = action_space
        # print(state_space, action_space)
        self.seed = random.seed(seed)
        self.step_t = 0                             # record iteration of learning
        self.batch_size = batch_size
        # self.target_update = target_update
        self.gamma = gamma
        self.device = device 

        # Q NN init
        self.q_nn_update = Q_NN(state_space, action_space, seed).to(device)
        self.q_nn_frozen = Q_NN(state_space, action_space, seed).to(device)
        self.optimizer = optim.Adam(self.q_nn_update.parameters(), lr = learn_rate)

        # memory buffer
        self.memory = memoryBuffer(buffer_size = buffer_size, batch_size = batch_size)
        self.isReplay = isReplay       # if the agent contains replay buffer   
        self.uniform = uniform         # if the agent utilizes uniform behavior policy 

    
    def step(self, state, action, reward, next_state, terminal):
        """ process of learning """

        if self.isReplay:
            # store memory to buffer
            self.memory.push(state, action, reward, next_state, terminal)
            if len(self.memory) < self.batch_size:
                return

            experience_batch = self.memory.sample()
        else:
            # no replay, just learn the previous one
            experience_batch = self.memory.Transition(state, action, reward, \
                            next_state, terminal)
        self.learn(experience_batch)

    def select_action(self, state, eps=0):
        """return the policy of self.q_nn_update """
        if self.uniform:
            return random.choice(np.arange(self.action_space))
        else:
            # epsilon-greedy
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_nn_update.eval()
            with torch.no_grad():
                action_values = self.q_nn_update(state)
            self.q_nn_update.train()

            if random.random() > eps:
                # best action
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_space))

    def learn(self, experience_batch):
        """ 
        Update Q hyperparameters from experience data
        input: experience batch data 
        """

        # if there is no replay, fit the data in the namedtuple format
        if self.isReplay:
            batch = self.memory.Transition(*zip(*experience_batch))
        else:  batch = experience_batch

        # if the value is not list, need .T for the numpy matrix convert
        state_batch = torch.from_numpy(np.matrix(batch.state)).float().to(self.device)
        action_batch = torch.from_numpy(np.matrix(batch.action).T).long().to(self.device)
        reward_batch = torch.from_numpy(np.matrix(batch.reward).T).float().to(self.device)
        next_state_batch = torch.from_numpy(np.matrix(batch.next_state)).float().to(self.device)
        terminal_batch = torch.from_numpy(np.matrix(batch.terminal).T).int().to(self.device)

        # get the max_b Q_frozen(next_state, b)
        # q_target_next_state.shape should be [BATCH_SIZE, 1]
        q_target_next_state = self.q_nn_frozen(next_state_batch).detach().max(1)[0].unsqueeze(1)
        # assert q_target_cur_state.shape[0] == self.memory.batch_size, raise RuntimeError("shape error")

        """ calculate 
        yt = reward                                         if terminal 
        yt = reward + gamma * max_b Q_frozen(next_state, b)
        """
        q_target_cur_state = reward_batch + self.gamma * q_target_next_state * (1 - terminal_batch)
        q_update_cur_state = self.q_nn_update(state_batch).gather(1, action_batch)
        # print(q_update_cur_state.shape, q_target_cur_state.shape)
        # gradient descent step
        loss = F.mse_loss(q_update_cur_state, q_target_cur_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_frozen_nn(self):
        # update frozen network with local network parameters

        # hard update
        for target_param, local_param in zip(self.q_nn_frozen.parameters(),
                                           self.q_nn_update.parameters()):
            target_param.data.copy_(local_param.data)

    def save_checkpoint(self, t, story):
        """ save the model as checkpoint given episode t """
        
        # save nn model
        filename = os.path.join(os.getcwd(), 'model', '%s_nn_model_%s.pth' % (story, t))
        torch.save(self.q_nn_update.state_dict(), filename)
