import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Score_Env(gym.Env):
    # Score environment main class
    def __init__(self, list_scores):
        self.num_score = -1
        self.list_scores = list_scores
        self.num_note = 0
        self.state = (self.list_scores[0][0], self.list_scores[0][1], 1)

        self.reset()

    def step(self, action_finger):
        # next step env
        current_finger = self.state

        # check states and action
        assert 1 <= action_finger + 1 <= 5
        reward = 0.0

        if current_finger == action_finger:
            reward += 10

        if current_finger != action_finger:
            reward += -10

        done = bool(self.num_note + 1 == 9)
        if not done:
            self.num_note += 1
            self.state = action_finger

        return np.array(self.state), reward, done, {}

    def next_key(self):
        # next keyboard key
        return self.list_scores[self.num_score][self.num_note + 1]

    def size_env(self):
        # number of notes
        return len(self.list_scores)

    def reset(self):
        # reset enviroment
        self.num_note = 0
        self.num_score = self.num_score + 1
        self.state = 1
        return np.array(self.state)
