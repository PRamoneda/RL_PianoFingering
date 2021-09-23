import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from music21 import articulations
from torch.autograd import Variable
from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd

import loader
from fingers_keyboard import Score_Env
import music21

EpisodeStats = namedtuple("Stats", ["episode_rewards", "episode_loss"])


def tt(ndarray):
    # return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def tt_bool(ndarray):
    return Variable(torch.from_numpy(ndarray).bool(), requires_grad=False)


def encode_states(state):
    current_finger, current_key, following_key = state
    encoded = np.zeros(125)
    index_encoded = int(current_finger + 5*current_key + 25*following_key)
    encoded[index_encoded] = 1.0
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
        self.fc1 = torch.nn.Linear(125, 5)
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

    def get_action(self, x, epsilon=-1.0):
        # encode state
        x_coded = encode_states(x)
        u = np.argmax(self._q(tt(x_coded)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes, epsilon):
        stats = EpisodeStats(episode_loss=np.zeros(episodes), episode_rewards=np.zeros(episodes))

        N = 25
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
                # Implement this
                self._replay_buffer.add_transition(s, a, ns, r, last)
                batch = self._replay_buffer.random_next_batch(BATCH_SIZE)

                x = np.zeros((len(batch), 125))
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

                stats.episode_loss[e] = loss


                self._q_optimizer.step()

                # update target network
                # HARD
                # if t % N == 0:
                #  hard_update(self._q_target, self._q)
                # SOFT
                soft_update(self._q_target, self._q, 0.1)

                s = ns
                count += 1

            if stats.episode_rewards[e] >= 2000:
                break

        return stats


def plot_episode_stats(stats, smoothing_window=10):
    # Plot the episode reward over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_loss)
    plt.xlabel("Episode")
    plt.ylabel("Episode Loss")
    plt.title("Episode Loss over Time")
    fig1.savefig('episode_Losses.png')
    plt.close(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    plt.close(fig2)


def save_binary(obj, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


if __name__ == "__main__":
    env = Score_Env(loader.load_test3(5000))

    dqn = DQN(gamma=0.99)

    epsilon = 0.15
    episodes = 500

    stats = dqn.train(episodes, epsilon)
    plot_episode_stats(stats)

    sc = music21.converter.parse('score.musicxml')
    rh = sc.parts[0].flat.getElementsByClass("Note")
    test = [loader.translateADAM1[a] for a in loader.ADAM1]
    print(test)
    env_test = Score_Env([test])
    s = (2, 2, 4)
    rh[0].articulations.append(articulations.Fingering(2 + 1))
    for ii in range(len(test)):
        a = dqn.get_action(s)
        rh[ii + 1].articulations.append(articulations.Fingering(a + 1))
        s, r, d, _ = env_test.step(a)
        if d:
            break

    rh.show()
