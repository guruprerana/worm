from enum import Enum
from typing import List
from gymnasium import spaces
import numpy as np
from miniworld.entity import Box
from agents.miniworld.riskyminiworld import RiskyMiniworld


class FairCollection(RiskyMiniworld):
    class Tasks(Enum):
        COLLECT_BOXES = "collect-boxes"

    def __init__(self, max_episode_steps=500, **kwargs):
        super().__init__(max_episode_steps=500, **kwargs)

        # Allow only movement actions
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.observation_space = spaces.Dict(
            {
                "image": self.observation_space, 
                "num_green_collected": spaces.Box(low=0.0, high=20.0, shape=(1,)),
                "num_blue_collected": spaces.Box(low=0.0, high=20.0, shape=(1,)),
                "num_difference": spaces.Box(low=-20.0, high=20.0, shape=(1,))
            }
        )

    def _gen_world(self):
        self.hall = self.add_rect_room(min_x=0, max_x=10, min_z=0, max_z=10)

        # load agent
        pos = None
        dir = None
        if self.set_env_state:
            pos = np.copy(self.set_env_state["agent"]["pos"])
            dir = self.set_env_state["agent"]["dir"]
            self.set_env_state = None
        elif self.init_states:
            state = self.np_random.choice(self.init_states, 1)[0]
            if state:
                pos = np.copy(state["agent"]["pos"])
                dir = state["agent"]["dir"]

        if pos is not None and dir is not None:
            self.place_agent(pos=pos, dir=dir)
        else:
            self.place_agent(min_x=0, max_x=4, min_z=0, max_z=4)

        # load boxes
        self.green_boxes: List[Box] = []
        self.blue_boxes: List[Box] = []
        for _ in range(12):
            box = Box("green", 0.4)
            self.green_boxes.append(box)
            self.place_entity(box)
        
        for _ in range(8):
            box = Box("blue", 0.4)
            self.blue_boxes.append(box)
            self.place_entity(box)

        self.num_green_picked = 0.0
        self.num_blue_picked = 0.0

    def get_env_state(self, **kwargs):
        return {
            "agent": {
                "pos": np.copy(self.agent.pos), 
                "dir": self.agent.dir,
            }
        }
    
    def get_loss_eval(self, **kwargs):
        return 0
    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["diff_colors"] = float(np.abs(self.num_green_picked - self.num_blue_picked))
        info["blue_picked"] = self.num_blue_picked
        info["green_picked"] = self.num_green_picked
        obs = self.augment_obs(obs)
        return obs, info
    
    def step(self, action):
        # self.previous_imbalance = float(np.abs(self.num_green_picked - self.num_blue_picked))
        obs, reward, termination, truncation, info = super(RiskyMiniworld, self).step(action)

        if self.agent.carrying:
            if self.agent.carrying in self.green_boxes:
                reward += float(np.exp(self.num_blue_picked - self.num_green_picked))
                self.num_green_picked += 1.0
            elif self.agent.carrying in self.blue_boxes:
                reward += float(np.exp(self.num_green_picked - self.num_blue_picked))
                self.num_blue_picked += 1.0

            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None

        info["env_state"] = self.get_env_state()
        info["loss_eval"] = self.get_loss_eval()
        info["diff_colors"] = float(np.abs(self.num_green_picked - self.num_blue_picked))
        info["blue_picked"] = self.num_blue_picked
        info["green_picked"] = self.num_green_picked
        obs = self.augment_obs(obs)
        reward += self.get_reward()
        termination = self.eval_terminated() or termination
        return obs, reward, termination, truncation, info
    
    def get_reward(self, **kwargs) -> float:
        # current_imbalance = float(np.abs(self.num_green_picked - self.num_blue_picked))
        # balance_improvement = self.previous_imbalance - current_imbalance
        # base_reward =  0.1 * (self.num_green_picked + self.num_blue_picked)
        return 0
    
    def eval_terminated(self, **kwargs) -> bool:
        return False
    
    def augment_obs(self, obs) -> dict:
        return {
            "image": obs,
            "num_green_collected": np.array([self.num_green_picked], dtype=np.float32),
            "num_blue_collected": np.array([self.num_blue_picked], dtype=np.float32),
            "num_difference": np.array([float(self.num_green_picked - self.num_blue_picked)], dtype=np.float32),
        }

    