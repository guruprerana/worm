import gymnasium as gym
from agents.miniworld.env1 import RiskyMiniworldEnv1
from agents.miniworld.faircollection import FairCollection
from agents.miniworld.boxrelay import BoxRelay

gym.register("RiskyMiniworldEnv1-v0", RiskyMiniworldEnv1)
gym.register("FairCollection-v0", FairCollection)
gym.register("BoxRelay-v0", BoxRelay)
