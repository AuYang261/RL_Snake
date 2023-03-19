# -*- coding = utf-8 -*-
# @Time : 2023/2/23 23:47
# @Author : AuYang
# @File : game.py
# @Software : PyCharm

import os
import numpy as np
import random
import pygame


class Board:
    """
    The positive of x-axis and y-axis of map coordinate is down and right
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.bodies = [np.array([rows // 2, cols // 2])]
        self.points = np.zeros((rows, cols), dtype=int)
        # 0: blank, 1: head, 2: body, 3: food
        self.points[self.bodies[0][0], self.bodies[0][1]] = 1
        self.availables = list(range(self.rows * self.cols))
        self.availables.remove(self.__xy2liner(self.bodies[0]))
        self.food_pos = self.__gen_food()
        # 0: down, 1: left, 2: up, 3: right
        self.direct = 0
        # running or end
        self.running = True
        self.smell_dist = 3
        self.die_reward = -5
        self.food_reward = 2
        self.score = 0
        self.steps = 0

    def __gen_food(self) -> np.array:
        liner = random.choice(self.availables)
        xy = self.__liner2xy(liner)
        self.points[xy[0], xy[1]] = 3
        return xy

    def __xy2liner(self, xy: np.array or list[np.array]) -> int or list[int]:
        return xy[0] * self.cols + xy[1]

    def __liner2xy(self, liner: int) -> np.array:
        return np.array([liner // self.cols, liner % self.cols])

    def __in_range(self, xy: np.array) -> bool:
        return 0 <= xy[0] < self.rows and 0 <= xy[1] < self.cols

    @staticmethod
    def __rotate90(vector: np.array, count: int) -> np.array:
        """
        rotate vector count*90 degrees clockwise
        """
        # clockwise to anticlockwise
        radian = -count * np.pi / 2
        c = round(np.cos(radian))
        s = round(np.sin(radian))
        return np.dot(vector, np.array([[c, s], [-s, c]])).astype(int)

    @staticmethod
    def __distance(xy1: np.array, xy2: np.array) -> float:
        return np.sqrt(np.sum(np.square(xy1 - xy2)))

    def one_step(self, action: int, relative: bool = True) -> float:
        """
        @return the reward
        """

        self.steps += 1
        if relative:
            if action == 1:
                # forward
                pass
            elif action == 2:
                # turn right
                self.direct = (self.direct + 1) % 4
            elif action == 3:
                # turn left
                self.direct = (self.direct - 1) % 4
            else:
                raise Exception("Unknown action {}".format(action))
        else:
            self.direct = action

        # forward direction is the positive of the x-axis in body coordinate,
        # which means self.direct is
        # the included angle of the forward direction and the x-axis in map coordinate(in 90 degrees units)
        next_head_pos = self.bodies[0] + self.__rotate90(np.array([1, 0]), self.direct)

        if not self.__in_range(next_head_pos) or self.points[next_head_pos[0], next_head_pos[1]] == 2:
            # out of range or colliding body
            self.running = False
            # return a minus reward
            return self.die_reward
        elif self.points[next_head_pos[0], next_head_pos[1]] == 0:
            # blank
            self.availables.remove(self.__xy2liner(next_head_pos))
            self.availables.append(self.__xy2liner(self.bodies[-1]))
            self.points[next_head_pos[0], next_head_pos[1]] = 1
            if len(self.bodies) > 1:
                self.points[self.bodies[0][0], self.bodies[0][1]] = 2
            else:
                self.points[self.bodies[0][0], self.bodies[0][1]] = 0
            self.points[self.bodies[-1][0], self.bodies[-1][1]] = 0
            self.bodies.insert(0, next_head_pos)
            self.bodies.pop()
            dist = self.__distance(self.bodies[0], self.food_pos)
            return (self.smell_dist - dist) / self.smell_dist if dist < self.smell_dist else 0
        elif self.points[next_head_pos[0], next_head_pos[1]] == 1:
            # head
            raise Exception("There are two heads in ({}, {}) and ({}, {})".format(*self.bodies[0], *next_head_pos))
        elif self.points[next_head_pos[0], next_head_pos[1]] == 3:
            # food
            self.availables.remove(self.__xy2liner(next_head_pos))
            # self.points[self.bodies[-1][0], self.bodies[-1][1]] = 0  # debug
            # self.availables.append(self.__xy2liner(self.bodies[-1]))  # debug
            self.points[next_head_pos[0], next_head_pos[1]] = 1
            self.points[self.bodies[0][0], self.bodies[0][1]] = 2
            self.bodies.insert(0, next_head_pos)
            # self.bodies.pop()  # debug
            self.food_pos = self.__gen_food()
            self.score += 1
            return self.food_reward
        else:
            raise Exception("Unknown state {} of point ({}, {})".format(self.points[next_head_pos[0], next_head_pos[1]],
                                                                        *next_head_pos))

    def get_state(self) -> tuple[int, int, bool, bool, bool, int]:
        """
        @return state consisted of food relative position and whether each direction(left, forward, right) is obstacle
        and the direction(left: 0, forward: 1, right: 2) closest to the obstacle
        The position is not normalized.
        The forward and the left is the positive direction of the x-axis and the y-axis
        """
        # coordinate of vector rotate in the opposite direction of the coordinate axis
        relative_pos = self.__rotate90(self.food_pos - self.bodies[0], -self.direct)
        adjacency = np.array(
            [self.bodies[0] + self.__rotate90(np.array(v), self.direct) for v in [[0, 1], [1, 0], [0, -1]]])
        closest = -1
        for i in range(1, np.min([self.rows, self.cols]) + 1):
            xy_pos = [self.bodies[0] + self.__rotate90(np.array(v), self.direct) for v in [[0, i], [i, 0], [0, -i]]]
            liner = [self.__xy2liner(xy) if self.__in_range(xy) else -1 for xy in xy_pos]
            obstacles = [(l not in self.availables) for l in liner]
            if any(obstacles):
                closest = np.argmax(obstacles)
                break
        assert closest != -1
        return (
            *relative_pos, *(self.__xy2liner(pos) not in self.availables for pos in adjacency), closest)

    def get_running(self) -> bool:
        return self.running

    def get_score(self) -> int:
        return self.score


class Game:

    def __init__(self, rows, cols, display=False):
        self.rows = rows
        self.cols = cols
        self.board = Board(rows, cols)
        self.display = display
        if self.display:
            pygame.init()
            self.window_size = (600, 600)
            self.block_size = (self.window_size[0] / self.cols, self.window_size[1] / self.rows)
            self.__screen = pygame.display.set_mode(self.window_size, 0, 32)
            pygame.display.set_caption('Snake')
            self.__ui_white = pygame.Surface(self.block_size)
            self.myfont = pygame.font.Font(None, 60)

    def draw(self):
        if not self.__screen:
            return
        self.__screen.fill(color='white')
        self.__ui_white.fill(color='red')
        self.__screen.blit(self.__ui_white,
                           (self.board.food_pos[1] * self.block_size[0], self.board.food_pos[0] * self.block_size[1]))
        for i, block in enumerate(self.board.bodies):
            if i == 0:
                self.__ui_white.fill(color='yellow')
            else:
                self.__ui_white.fill(color='blue')
            self.__screen.blit(self.__ui_white,
                               (block[1] * self.block_size[0], block[0] * self.block_size[1]))
        score_text = self.myfont.render('Score: {}'.format(self.board.score), True, (0, 0, 0))
        self.__screen.blit(score_text, (0, 0))
        for event in pygame.event.get():
            pass
        pygame.display.update()

    def restart(self):
        self.board = Board(self.rows, self.cols)

    def __del__(self):
        pygame.quit()


if __name__ == '__main__':
    game = Game(10, 10, True)
    while game.board.get_running():
        print(game.board.get_state())
        print(game.board.points)
        action = -1
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    action = 0
                elif event.key == pygame.K_a:
                    action = 1
                elif event.key == pygame.K_w:
                    action = 2
                elif event.key == pygame.K_d:
                    action = 3
        if action != -1:
            game.board.one_step(action, False)
        else:
            continue
        game.draw()
    print(game.board.score)
