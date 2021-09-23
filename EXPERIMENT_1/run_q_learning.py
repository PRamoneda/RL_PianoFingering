import pickle
import sys
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from music21 import articulations
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd
from fingers_keyboard import Score_Env
import music21

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def tt_bool(ndarray):
    return Variable(torch.from_numpy(ndarray).bool(), requires_grad=False)


def encode_states(state):
    current_finger = state
    encoded = np.zeros(5)
    encoded[current_finger] = 1.
    return encoded


def soft_update(target, source, tau):
    for pt, ps in zip(list(target.parameters()), list(source.parameters())):
        pt.data = (1 - tau) * pt.data + tau * ps.data


def hard_update(target, source):
    for pt, ps in zip(list(target.parameters()), list(source.parameters())):
        pt.data = ps.data


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        # input: (one-hot) 88, (one-hot) 88, (one-hot) 5
        self.fc1 = torch.nn.Linear(5, 5)
        self.sf = torch.nn.Softmax(dim=0)
        # output: (one-hot) 5

    def forward(self, x):
        x = self.fc1(x)
        return self.sf(x)


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_size):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])
        self._size = 0
        self._max_size = max_size

    def add_transition(self, state, action, ns, reward, done):
        state_encoded = encode_states(state)
        ns_encoded = encode_states(ns)
        if self._size < self._max_size:
            self._data.states.append(state_encoded)
            self._data.actions.append(action)
            self._data.next_states.append(ns_encoded)
            self._data.rewards.append(reward)
            self._data.terminal_flags.append(done)
            self._size += 1
        else:
            self._data.states = self._data.states[1:] + [state_encoded]
            self._data.actions = self._data.actions[1:] + [action]
            self._data.next_states = self._data.next_states[1:] + [ns_encoded]
            self._data.rewards = self._data.rewards[1:] + [reward]
            self._data.terminal_flags = self._data.terminal_flags[1:] + [done]

    def random_next_batch(self, batch_size):
        indexes = np.random.permutation(self._size)[:batch_size]

        result = []
        for i in indexes:
            result.append(
                [self._data.states[i], self._data.actions[i], self._data.next_states[i], self._data.rewards[i],
                 self._data.terminal_flags[i]])

        return result


class DQN:
    def __init__(self, gamma):
        self._q = Q()
        self._q_target = Q()

        self._gamma = gamma
        self._loss_function = nn.MSELoss()
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = 5

        self._replay_buffer = ReplayBuffer(1e6)

    def get_action(self, x, epsilon):
        # encode state
        x_coded = encode_states(x)
        u = np.argmax(self._q(tt(x_coded)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes, epsilon):
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        BATCH_SIZE = 32
        count = 0
        for e in range(episodes):
            print("%s/%s" % (e + 1, episodes))
            s = env.reset()
            last = False
            while not last:
                a = self.get_action(s, epsilon)
                ns, r, last, _ = env.step(a)

                stats.episode_rewards[e] += r
                stats.episode_lengths[e] = count

                self._replay_buffer.add_transition(s, a, ns, r, last)
                batch = self._replay_buffer.random_next_batch(BATCH_SIZE)

                x = np.zeros((len(batch), 5))
                a = np.zeros((len(batch), 5), bool)
                y = np.zeros(len(batch))
                # 0: state, 1: action, 2: next_state, 3: reward, 4: terminal_flag
                for j in range(len(batch)):
                    x[j] = batch[j][0]
                    a[j][batch[j][1]] = True
                    if batch[j][4]:
                        y[j] = batch[j][3]
                    else:
                        # normal q-learning
                        # y[j] = batch[j][3] + self._gamma * np.amax(self._q_target(tt(batch[j][2])).cpu().detach().numpy())
                        # double q-learning
                        y[j] = batch[j][3] + self._gamma * self._q_target(tt(batch[j][2])).cpu().detach().numpy()[
                            np.argmax(self._q(tt(batch[j][2])).cpu().detach().numpy())]

                self._q_optimizer.zero_grad()

                # update parameters
                q_sa = torch.masked_select(self._q(tt(x)), tt_bool(a))
                loss = self._loss_function(q_sa, tt(y))
                loss.backward()
                self._q_optimizer.step()

                # update target network
                # HARD
                # if t % N == 0:
                #  hard_update(self._q_target, self._q)
                # SOFT
                soft_update(self._q_target, self._q, 0.1)

                s = ns
                count += 1

        return stats


def save_binary(obj, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data

KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'x': None}

def parse_note(note):
    n = KEY_TO_SEMITONE[note[:-1].lower()]
    octave = int(note[-1]) + 1
    return octave * 12 + n - 21


if __name__ == "__main__":
    env = Score_Env([58*8]*1000)

    dqn = DQN(gamma=0.99)

    episodes = 2000
    epsilon = 0.3

    stats = dqn.train(episodes, epsilon)

    sc = music21.converter.parse('test1.mxl')
    rh = sc.parts[0].flat.getElementsByClass("Note")
    test = [parse_note(str(note.pitch)) for note in rh]
    print(test)
    env_test = Score_Env([test])
    s = 1
    rh[0].articulations.append(articulations.Fingering(1 + 1))
    for ii in range(len(test)):
        a = dqn.get_action(s, epsilon)
        rh[ii + 1].articulations.append(articulations.Fingering(a + 1))
        s, _, d, _ = env_test.step(a)
        if d:
            break

    rh.show()
