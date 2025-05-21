from enum import Enum
from typing import Any, List, Optional
import gymnasium as gym
import numpy as np
import copy

from miniworld.entity import Entity
from miniworld.miniworld import MiniWorldEnv

class RiskyMiniworld(MiniWorldEnv):
    def __init__(
            self, 
            init_states: Optional[List[Any]]=None,
            task_str: Optional[str]=None,
            **kwargs
        ):
        self.init_states=init_states
        self.task_str = task_str
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        if options and isinstance(options, dict):
            if "state" in options and options["state"] != None:
                self.set_env_state = options["state"]
        else:
            self.set_env_state = None
        obs, info = super().reset(seed=seed, options=options)
        info["env_state"] = self.get_env_state()
        info["loss_eval"] = self.get_loss_eval()
        return obs, info
    
    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        termination = self.eval_terminated() or termination
        reward += self.get_reward(termination=termination, truncation=truncation)
        info["env_state"] = self.get_env_state()
        info["loss_eval"] = self.get_loss_eval(termination=termination, truncation=truncation)
        return obs, reward, termination, truncation, info

    def get_env_state(self, **kwargs) -> Any:
        raise NotImplementedError
    
    def get_loss_eval(self, **kwargs) -> float:
        raise NotImplementedError
    
    def get_reward(self, **kwargs) -> float:
        raise NotImplementedError
    
    def eval_terminated(self, **kwargs) -> bool:
        raise NotImplementedError

