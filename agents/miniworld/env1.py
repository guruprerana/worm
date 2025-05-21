from enum import Enum
import numpy as np
from gymnasium import spaces
from conformal.miniworld.riskyminiworld import RiskyMiniworld


class RiskyMiniworldEnv1(RiskyMiniworld):
    class Tasks(Enum):
        GOTO_MIDDLE_BOTTOM_ENTRY = "goto-middle-bottom-entry"
        GOTO_MIDDLE_TOP_ENTRY = "goto-middle-top-entry"
        GOTO_MIDDLE_BOTTOM_EXIT = "goto-middle-bottom-exit"
        GOTO_MIDDLE_TOP_EXIT = "goto-middle-top-exit"
        GOTO_RIGHT_HALL = "goto-right-hall"

    def __init__(self, max_episode_steps=300, **kwargs):
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        # Allow only movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.left_hall = self.add_rect_room(min_x=0, max_x=5, min_z=0, max_z=21)
        self.middle_bottom = self.add_rect_room(min_x=6, max_x=16, min_z=0, max_z=10)
        self.middle_top = self.add_rect_room(min_x=6, max_x=16, min_z=11, max_z=21)
        self.right_hall = self.add_rect_room(min_x=17, max_x=22, min_z=0, max_z=21)

        self.connect_rooms(self.left_hall, self.middle_bottom, min_z=4, max_z=6)
        self.connect_rooms(self.left_hall, self.middle_top, min_z=15, max_z=17)
        self.connect_rooms(self.middle_bottom, self.right_hall, min_z=4, max_z=6)
        self.connect_rooms(self.middle_top, self.right_hall, min_z=15, max_z=17)

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
            self.place_agent(min_x=2, max_x=4, min_z=8, max_z=10)

    def get_env_state(self, **kwargs):
        return {
            "agent": {
                "pos": np.copy(self.agent.pos), 
                "dir": self.agent.dir,
            }
        }
    
    def get_reward(self, **kwargs):
        target_state = self.get_target_state()
        agent_pos = np.array(self.agent.pos)
        dist = np.linalg.norm(agent_pos - target_state)
        
        # Direction component
        direction_to_target = target_state - agent_pos
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
        agent_direction = self.agent.dir_vec
        direction_alignment = np.dot(direction_to_target, agent_direction)
        
        # Combined reward
        distance_reward = -0.1 * dist
        direction_reward = 0.05 * max(0, direction_alignment)
        
        if dist <= 0.3:
            success_reward = 10.0
        else:
            success_reward = 0.0
            
        return distance_reward + direction_reward + success_reward
    
    def get_loss_eval(self, **kwargs):
        return 0
    
    def eval_terminated(self, **kwargs):
        target_state = self.get_target_state()
        dist = np.linalg.norm(np.array(self.agent.pos) - target_state)
        if dist <= 0.3:
            return True
        return False
    
    def get_target_state(self):
        if self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY:
            return np.array([5.5, 0, 5])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_TOP_ENTRY:
            return np.array([5.5, 0, 15])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_BOTTOM_EXIT:
            return np.array([16.5, 0, 5])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_MIDDLE_TOP_EXIT:
            return np.array([16.5, 0, 15])
        elif self.task_str == RiskyMiniworldEnv1.Tasks.GOTO_RIGHT_HALL:
            return np.array([19, 0, 10])
        else:
            raise ValueError
        
