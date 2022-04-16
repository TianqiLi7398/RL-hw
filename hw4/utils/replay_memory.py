from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))
Transition_obj = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'acc_obj'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

class Memory_new(Memory):

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition_obj(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition_obj(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition_obj(*zip(*random_batch))