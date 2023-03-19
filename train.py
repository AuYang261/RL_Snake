# -*- coding = utf-8 -*-
# @Time : 2023/2/24 21:15
# @Author : AuYang
# @File : train.py
# @Software : PyCharm
import random

import numpy as np
import os
import game
import time
import matplotlib.pyplot as plt


class Train:

    def __init__(self):
        if os.path.exists('best_15_15.npy'):
            self.state_action_value = np.load(
                'best_15_15.npy', allow_pickle=True).item()
        else:
            self.state_action_value = {}
        self.episodes = 100000
        self.rows = 15
        self.cols = 15
        self.display = False
        self.game = game.Game(self.rows, self.cols, self.display)
        self.alpha = 0.9
        self.gamma = 0.3
        self.epsilon = 0.01

    def train(self):
        rewards = []
        scores = []
        for episode in range(self.episodes):
            s = self.game.board.get_state()
            rewards.append(0)
            while self.game.board.get_running():
                if s not in self.state_action_value:
                    # optimistic initialize
                    self.state_action_value[s] = np.array([0., 0., 0.])
                action_value = self.state_action_value[s]
                # epsilon = self.epsilon
                epsilon = self.epsilon * np.exp(-episode / 100000)
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = np.argmax(action_value)
                reward = self.game.board.one_step(action + 1)  # 0~2 -> 1~3
                next_state = self.game.board.get_state()
                if next_state not in self.state_action_value:
                    self.state_action_value[next_state] = np.array(
                        [0., 0., 0.])
                # Q-learning
                # self.state_action_value[s][action] += self.alpha * (reward + self.gamma * np.max(
                #     self.state_action_value[next_state]) - self.state_action_value[s][action])
                # Expectation Sarsa
                prob = np.array([0., 0., 0.])
                prob[np.argmax(self.state_action_value[next_state])] = 1.0
                self.state_action_value[s][action] += self.alpha * (reward + self.gamma * np.sum(
                    self.state_action_value[next_state] * (epsilon + (1 - epsilon) * prob)) -
                    self.state_action_value[s][action])
                s = next_state
                # print(s)
                rewards[-1] += reward
                if self.display:
                    time.sleep(0.05)
                    self.game.draw()
            scores.append(self.game.board.get_score())
            print(scores[-1], self.game.board.steps, rewards[-1], episode)
            self.game.restart()
            # print(self.state_action_value)
            if episode and episode % 1000 == 0:
                np.save('state_action_value.npy', self.state_action_value)
                plt.scatter((range(episode + 1)), scores, s=1)
                plt.draw()
                plt.savefig("imgs/rewards_{}.png".format(episode))


if __name__ == '__main__':
    train = Train()
    train.train()
    pass
