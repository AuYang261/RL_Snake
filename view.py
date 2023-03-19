# -*- coding = utf-8 -*-
# @Time : 2023/3/19 19:43
# @Author : AuYang
# @File : view.py.py
# @Software : PyCharm

import random

import numpy as np
import os
import sys

import pygame

import game
import time
import matplotlib.pyplot as plt


class View:

    def __init__(self, model: str):
        self.state_action_value = np.load(model, allow_pickle=True).item()
        self.rows = 15
        self.cols = 15
        self.display = True
        self.fps = 20
        self.game = game.Game(self.rows, self.cols, self.display)
        self.alpha = 0.9
        self.gamma = 0.3
        self.epsilon = 0.0

    def run(self):
        while self.game.board.get_running():
            s = self.game.board.get_state()
            if s not in self.state_action_value:
                # optimistic initialize
                self.state_action_value[s] = np.array([0., 0., 0.])
            action_value = self.state_action_value[s]
            action = np.argmax(action_value)
            reward = self.game.board.one_step(action + 1)  # 0~2 -> 1~3

            if self.display:
                time.sleep(1 / self.fps)
                self.game.draw()


if __name__ == '__main__':
    view = View('best_15_15.npy')
    view.run()
    sys.exit()
