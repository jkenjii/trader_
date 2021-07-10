import random

from collections import namedtuple
from typing import List
import numpy as np


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done'])


class ReplayMemory(object):

    def __init__(self, capacity: int, transition_type: namedtuple = Transition):
        self.capacity = capacity
        self.Transition = transition_type

        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
            #print(self.Transition(*args))
            #print(len(self.memory))
        else:
            self.memory.pop(0)
            self.memory.append(self.Transition(*args))
            #self.memory[self.position] = self.Transition(*args)

        self.position = (self.position + 1) % self.capacity
    
    def pop(self)-> List[namedtuple]:
        #print("HAKUBA")
        #print("antes",self.memory[-1])
        self.memory.pop()
        #print("depois", self.memory[-1])

    def sample(self, batch_size) -> List[namedtuple]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

    def sample_DRQN_errado(self, batch_size, trace_length = 6) -> List[namedtuple]:
        #print(self.memory)
        print(self.position)
        indices = []
        sample = []
        #print(sample)
        while len(indices) < batch_size:
            index = random.randint(trace_length, len(self.memory)-1) 
            #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)#pode repetir indice
            print("OLHA AQUI", ids           )
            history = self.memory[index - trace_length:index]
            #print(history)
            #print(len(indices))
            for item in history:
                #print(item)
                sample.append(item) 
            #print("final sample", sample)
            indices.append(index)
              
              
        return sample
        
    def sample_DRQN(self, batch_size, trace_length = 4) -> List[namedtuple]:
        #print(self.memory)
        print(self.position)
        indices = []
        sample = []
        #print(sample)
        while len(indices) < batch_size:
            index = random.randint(trace_length, len(self.memory)-1) 
            #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)
            #print("OLHA AQUI", ids           )
            history = self.memory[index - trace_length:index]
            #print(history)
            #print(len(indices))
            for item in history:
                #print(item)
                sample.append(item) 
            #print("final sample", sample)
            indices.append(index)
              
        #print(sample)     
        return sample

    def sample_DRQN_pos_DODO_gambia(self,  trace_length ) -> List[namedtuple]:
        #print(self.memory)
        #print(self.position)
        indices = []
        sample = []
        #print(sample)
        
        index = random.randint(trace_length, len(self.memory)-1) 
        #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)
        #print("OLHA AQUI", ids           )
        history = self.memory[index - trace_length:index]
        #print(history)
        #print(len(indices))
        for item in history:
            #print(item)
            sample.append(item) 
            #print("final sample", sample)
        indices.append(index)
              
        #print(sample)     
        return sample



