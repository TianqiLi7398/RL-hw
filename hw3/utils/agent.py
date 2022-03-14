import gym
import random
import numpy as np
from learner import Q_NN, memoryBuff
import torch
import torch.optim as optim
import torch.nn.functional as F

class Agent:

    def __init__(self, state_space, action_space, 
            seed = 0,
            device = 'cpu', 
            learn_rate = 5e-4,
            gamma = 0.99, 
            batch_size = 10,
            target_update = 10,
            buffer_size = int(1e5)):
        
        # parameter definition
        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(seed)
        self.step_t = 0                             # record iteration of learning
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.device = device 

        # Q NN init
        self.q_nn_update = Q_NN(state_space, action_space, seed).to(device)
        self.q_nn_frozen = Q_NN(state_space, action_space, seed).to(device)
        self.optimizer = optim.Adam(self.q_nn_update.parameters(), lr = learn_rate)

        # memory buffer
        self.memory = memoryBuff(buffer_size = buffer_size, batch_size = batch_size)

    
    def step(self, state, action, reward, next_state, terminal):
        """ process of learning """
        # store memory to buffer
        self.memory.push(state, action, reward, next_state, terminal)
        if len(self.memory) < self.batch_size:
            return

        # experience replay
        # self.step_t = (self.step_t + 1) % self.target_update
        # if self.step_t == 0 and len(self.memory) > self.batch_size:
        experience_batch = self.memory.sample()
        self.learn(experience_batch)

    def select_action(self, state, eps=0):
        """return the policy of self.q_nn_update """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_nn_update.eval()
        with torch.no_grad():
            action_values = self.q_nn_update(state)
        self.q_nn_update.train()

        # epsilon-greedy
        if random.random() > eps:
            # best action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arrange(self.action_space))

    def learn(self, experience_batch):

        batch = self.memory.Transition(*zip(experience_batch))
        state_batch = torch.from_numpy(np.matrix(batch.state).T).float().to(self.device)
        action_batch = torch.from_numpy(np.matrix(batch.action).T).long().to(self.device)
        reward_batch = torch.from_numpy(np.matrix(batch.reward).T).float().to(self.device)
        next_state_batch = torch.from_numpy(np.matrix(batch.next_state).T).float().to(self.device)
        terminal_batch = torch.from_numpy(np.matrix(batch.terminal).T).int().to(self.device)

        # get the max_b Q_frozen(next_state, b)
        q_target_next_state = self.q_nn_frozen(next_state_batch).detach().max(1)[0].unsqueeze(1)
        """ calculate 
        yt = reward                                         if terminal 
        yt = reward + gamma * max_b Q_frozen(next_state, b)
        """
        q_target_cur_state = reward_batch + self.gamma * q_target_next_state * (1 - terminal_batch)
        q_update_cur_state = self.q_nn_update(state_batch).gather(1, action_batch)

        # gradient descent step
        loss = F.mse_loss(q_update_cur_state, q_target_cur_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.soft_update(self.q_nn_update, self.q_nn_frozen)

    def update_frozen_nn(self):
        # update frozen network

        # hard update
        for target_param, local_param in zip(self.q_nn_frozen.parameters(),
                                           self.q_nn_update.parameters()):
            target_param.data.copy_(local_param.data)