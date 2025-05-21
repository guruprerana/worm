from enum import Enum
import numpy as np
from gymnasium import spaces
from miniworld.entity import Box
from conformal.miniworld.riskyminiworld import RiskyMiniworld


class BoxRelay(RiskyMiniworld):
    class Tasks(Enum):
        GOTO_LEFT_HALL_TARGET = "goto-left-hall-target"
        GOTO_MIDDLE_BOTTOM_ENTRY = "goto-middle-bottom-entry"
        GOTO_MIDDLE_BOTTOM_TARGET = "goto-middle-bottom-target"
        GOTO_MIDDLE_BOTTOM_EXIT = "goto-middle-bottom-exit"
        GOTO_RIGHT_HALL_TARGET = "goto-right-hall-target"

        GOTO_MIDDLE_TOP_ENTRY = "goto-middle-top-entry"
        GOTO_MIDDLE_TOP_TARGET = "goto-middle-top-target"
        GOTO_MIDDLE_TOP_EXIT1 = "goto-middle-top-exit1"
        GOTO_MIDDLE_TOP_EXIT2 = "goto-middle-top-exit2"
        GOTO_RIGHT_HALL_TARGET_FROM_EXIT1 = "goto-right-hall-target-exit1"
        GOTO_RIGHT_HALL_TARGET_FROM_EXIT2 = "goto-right-hall-target-exit2"


    def __init__(self, max_episode_steps=300, **kwargs):
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

        self.action_space = spaces.Discrete(self.actions.move_back + 1)

    def agent_carries(self, ent):
        """
        Adds entity and makes agent carry it
        """
        self.agent.carrying = ent
        while True:
            ent_pos = self._get_carry_pos(self.agent.pos, ent)
            if not self.intersect(ent, ent_pos, ent.radius):
                break
            self.init_agent()
        ent.pos = ent_pos
        ent.dir = self.agent.dir
        self.entities.append(ent)

    def _gen_world(self):
        self.left_hall = self.add_rect_room(min_x=0, max_x=7, min_z=0, max_z=21)
        self.middle_bottom = self.add_rect_room(min_x=7.5, max_x=14.5, min_z=0, max_z=5)
        self.middle_top = self.add_rect_room(min_x=7.5, max_x=14.5, min_z=6, max_z=21)
        self.right_hall = self.add_rect_room(min_x=15, max_x=22, min_z=0, max_z=21)

        # only create doors when needed
        if self.task_str in (BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY, BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET):
            self.connect_rooms(self.left_hall, self.middle_bottom, min_z=1, max_z=4)
        elif self.task_str in (BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY, BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET):
            self.connect_rooms(self.left_hall, self.middle_top, min_z=12, max_z=15)
        elif self.task_str in (BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT, BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET):
            self.connect_rooms(self.middle_bottom, self.right_hall, min_z=1, max_z=4)
        elif self.task_str in (BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1, BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1):
            self.connect_rooms(self.middle_top, self.right_hall, min_z=9, max_z=12)
        elif self.task_str in (BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2, BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2):
            self.connect_rooms(self.middle_top, self.right_hall, min_z=15, max_z=18)

        self._gen_static_data()
        pos = None
        dir = None
        self.prev_carry_time = None
        if self.set_env_state:
            pos = np.copy(self.set_env_state["agent"]["pos"])
            dir = self.set_env_state["agent"]["dir"]
            self.prev_carry_time = self.set_env_state["carry_time"]
            self.set_env_state = None
        elif self.init_states:
            state = self.np_random.choice(self.init_states, 1)[0]
            if state:
                pos = np.copy(state["agent"]["pos"])
                dir = state["agent"]["dir"]
                self.prev_carry_time = state["carry_time"]

        if pos is not None and dir is not None:
            if self.intersect(self.agent, pos, self.agent.radius):
                self.init_agent()
            else:
                self.place_agent(pos=pos, dir=dir)
        else:
            self.init_agent()

        carry_box = Box("green", size=0.2)
        target_box = Box("blue")

        # need following line from line 581 of miniworld.py which is unfortunately called after gen_world
        self.max_forward_step = self.params.get_max("forward_step")
        if self.task_str == BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET:
            self.place_entity(target_box, room=self.left_hall, max_x=4)
        elif self.task_str in (
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY, 
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2,
        ):
            self.agent_carries(carry_box)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET:
            self.agent_carries(carry_box)
            self.place_entity(target_box, room=self.middle_bottom)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET:
            self.agent_carries(carry_box)
            self.place_entity(target_box, room=self.middle_top)
        elif self.task_str in (
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET, 
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1, 
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2
        ):
            self.agent_carries(carry_box)
            self.place_entity(target_box, room=self.right_hall, min_x=18)
        else:
            raise ValueError

        self.carry_box = carry_box
        self.target_box = target_box

        if self.prev_carry_time and self.task_str in (
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2,
        ):
            self.carry_time = self.prev_carry_time
        else:
            self.carry_time = 0.0

    def get_env_state(self, **kwargs):
        self.carry_time += 1
        return {
            "agent": {
                "pos": np.copy(self.agent.pos), 
                "dir": self.agent.dir,
            },
            "carry_time": self.carry_time,
        }

    def get_reward(self, termination=False, **kwargs):
        target_state = self.get_target_state()
        agent_pos = np.array(self.agent.pos)
        dist = np.linalg.norm(agent_pos - target_state)
        
        # Direction component
        direction_to_target = target_state - agent_pos
        direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
        agent_direction = self.agent.dir_vec
        direction_alignment = np.dot(direction_to_target, agent_direction)
        
        # Combined reward
        distance_reward = -0.1 * (1 - np.exp(-0.5 * dist))
        direction_reward = 0.05 * max(0, direction_alignment)
        
        success_reward = 0.0
        if termination:
            success_reward = 10.0

        time_penalty = -0.01
            
        return distance_reward + direction_reward + success_reward + time_penalty
    
    def get_loss_eval(self, termination=False, truncation=False, **kwargs):
        if self.task_str in (
            BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET, 
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY, 
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2,
        ):
            if truncation:
                return np.inf
            return -np.inf
        elif self.task_str in (
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET, 
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2,
        ):
            if termination:
                return self.carry_time
            if truncation:
                return np.inf
            return -np.inf
        else:
            raise ValueError
    
    def eval_terminated(self, **kwargs):
        target_state = self.get_target_state()
        dist = np.linalg.norm(np.array(self.agent.pos) - target_state)
        if self.task_str in (
            BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET,
        ):
            if dist <= 1.5:
                return True
        elif self.task_str in (
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET, 
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2,
        ):
            dist_carrying = np.linalg.norm(np.array(self.carry_box.pos) - target_state)
            if dist_carrying <= 1.5:
                return True
        else:
            if dist <= 0.3:
                return True
        return False
    
    def get_target_state(self):
        if self.task_str == BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET:
            return np.array(self.target_box.pos)
        elif self.task_str in (
            BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET, 
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
            BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1,
            BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2,
        ):
            return np.array(self.target_box.pos)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY:
            return np.array([7.25, 0, 2.5])
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT:
            return np.array([14.75, 0, 2.5])
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY:
            return np.array([7.25, 0, 13.5])
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1:
            return np.array([14.75, 0, 10.5])
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2:
            return np.array([14.75, 0, 16.5])
        else:
            raise ValueError
        
    def init_agent(self):
        if self.task_str == BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET:
            self.place_agent(room=self.left_hall)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY:
            self.place_agent(room=self.left_hall)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET:
            self.place_agent(min_x=6.9, max_x=7.6, min_z=1.0, max_z=4.0)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT:
            self.place_agent(room=self.middle_bottom)
        elif self.task_str == BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET:
            self.place_agent(min_x=14.4, max_x=15.1, min_z=1.0, max_z=4.0)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY:
            self.place_agent(room=self.left_hall)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET:
            self.place_agent(min_x=6.9, max_x=7.6, min_z=12.0, max_z=15.0)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1:
            self.place_agent(room=self.middle_top)
        elif self.task_str == BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2:
            self.place_agent(room=self.middle_top)
        elif self.task_str == BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1:
            self.place_agent(min_x=14.4, max_x=15.1, min_z=9.0, max_z=12.0)
        elif self.task_str == BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2:
            self.place_agent(min_x=14.4, max_x=15.1, min_z=15.0, max_z=18.0)
        else:
            raise ValueError


spec_graph = [
    {
        1: BoxRelay.Tasks.GOTO_LEFT_HALL_TARGET,
    },
    {
        2: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_ENTRY,
        6: BoxRelay.Tasks.GOTO_MIDDLE_TOP_ENTRY,
    },
    {
        3: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_TARGET,
    },
    {
        4: BoxRelay.Tasks.GOTO_MIDDLE_BOTTOM_EXIT,
    },
    {
        5: BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET,
    },
    {},
    {
        7: BoxRelay.Tasks.GOTO_MIDDLE_TOP_TARGET,
    },
    {
        8: BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT1,
        9: BoxRelay.Tasks.GOTO_MIDDLE_TOP_EXIT2,
    },
    {
        5: BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT1,
    },
    {
        5: BoxRelay.Tasks.GOTO_RIGHT_HALL_TARGET_FROM_EXIT2,
    }
]
