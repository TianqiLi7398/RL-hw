import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

class Q_NN(nn.Module):
    def __init__(self, action_space, state_space, seed):
        """ 
        Q network 
        """

        super(Q_NN, self).__init__()
        # build a NN 
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_space, 64)
        self.layer2 = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, action_space)



    def forward(self, state):
        value = self.layer1(state)        
        value = F.relu(value)
        value = self.layer2(value)
        value = F.relu(value)
        return self.layer3(value)

class memoryBuff:

    """ A uniform memory replay buffer """

    def __init__(self, buffer_size = int(1e5), batch_size = 10):
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))

    def push(self, *args):
        # self.Transition = (state, action, reward, next_state, terminal)
        # self.memory.append(experience)
        self.memory.append(self.Transition(*args))

    def sample(self):
        """ samples a batch of transitions """
        trans = random.sample(self.memory, self.batch_size)

        return trans
        

    def __len__(self):
        return len(self.memory)
