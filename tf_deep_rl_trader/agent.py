
import numpy as np

from abc import ABCMeta, abstractmethod

class Agent(metaclass=ABCMeta):    

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        
        raise NotImplementedError()
