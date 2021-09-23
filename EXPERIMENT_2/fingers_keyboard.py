import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Score_Env(gym.Env):
    def __init__(self, list_scores):
        self.num_score = -1
        self.list_scores = list_scores
        self.num_note = 0
        # self.state = (current_finger, current_note, finger_1, note_1, finger_2, note_2)
        self.state = (0, 0, 1)

        self.reset()

    def step(self, action_finger):
        current_finger, current_key, following_key = self.state

        # check states and action
        assert 1 <= action_finger + 1 <= 5
        assert 0 <= current_key < 5
        assert 0 <= following_key < 5
        reward = 0.0
        cond1, cond2, cond3, cond4, cond5 = False, False, False, False, False

        # Si hay un cruce de dedos
        if (current_key < self.next_key() and current_finger > action_finger) or \
           (current_key > self.next_key() and current_finger < action_finger):
            reward += -20
        else:
            cond1 = True

        # si se toca con un dedo que se ha usado para tocar una nota distinta
        if current_key != self.next_key() and current_finger == action_finger:
            reward += -20
        else:
            cond2 = True

        if current_key == self.next_key() and current_finger != action_finger:
            reward += -20
        else:
            cond3 = True

        # dedos libertah
        if abs(current_finger - action_finger) > abs(current_key - self.next_key()):
            reward = -20
        else:
            cond4 = True

        # combinaciones imposibles
        descendente = bool(current_key > self.next_key())
        ascendente = bool(current_key < self.next_key())
        if (descendente and current_finger in [0, 1, 2] and action_finger == 4) or \
           (descendente and current_finger == 2 and action_finger == 3) or \
           (ascendente and current_finger == 4 and action_finger in [0, 1, 2]) or\
           (ascendente and current_finger == 3 and action_finger == 2):

            reward += -100
        else:
            cond5 = True

        if cond1 and cond2 and cond3 and cond4 and cond5:
            reward += 50


        done = bool(self.num_note + 1 == len(self.list_scores[self.num_score]) - 1)
        if not done:
            self.num_note += 1
            self.state = (action_finger, following_key, self.next_key())

        return np.array(self.state), reward, done, {}

    def next_key(self):
        return self.list_scores[self.num_score][self.num_note + 1]

    def size_env(self):
        return len(self.list_scores)

    def reset(self):
        self.num_note = 0
        self.num_score = self.num_score + 1
        self.state = (0, 0, 1)
        return np.array(self.state)