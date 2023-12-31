import math

import pygame
import numpy as np

from nn import NeuralNetwork
from config import CONFIG


class Player():

    def __init__(self, mode, control=False):

        self.control = control  # if True, playing mode is activated. else, AI mode.
        self.pos = [100, 275]  # position of the agent
        self.direction = -1  # if 1, goes upwards. else, goes downwards.
        self.v = 0  # vertical velocity
        self.g = 9.8  # gravity constant
        self.mode = mode  # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists, camera, events=None):

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control
        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(mode, box_lists, agent_position, self.v)

        # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    def init_network(self, mode):

        # you can change the parameters below

        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [6, 20, 1]
        elif mode == 'helicopter':
            layer_sizes = [6, 20, 1]
        elif mode == 'thrust':
            layer_sizes = [6, 20, 1]
        return layer_sizes

    # def think(self, mode, box_lists, agent_position, velocity):
    #     i = 0
    #     while mode == 'helicopter':
    #         if len(box_lists):
    #             if box_lists[i].x > agent_position[0]:
    #                 x0 = agent_position[0] - box_lists[i].x
    #                 y0 = agent_position[1] - box_lists[i].gap_mid
    #                 if len(box_lists) > 1:
    #                     x1 = agent_position[0] - box_lists[i + 1].x
    #                     y1 = agent_position[1] - box_lists[i + 1].gap_mid
    #                 else:
    #                     x1 = box_lists[i].x + 200
    #                     y1 = y0
    #                 break
    #             else:
    #                 i += 1
    #         else:
    #             x0 = x1 = 1280
    #             y0 = y1 = 360
    #             break
    #     z1 = math.sqrt((x0 ** 2) + (y0 ** 2))
    #
    #     # nn_input = [[velocity / 10], [z1 / 1468.6],  [x0 / 1280],
    #     #             [y0 / 720], [x1 / 1280], [y1 / 720]] best result till now
    #
    #     nn_input = [[velocity / 10], [z1 / 1468.6], [x0 / 1280],
    #                 [y0 / 720], [x1 / 1280], [y1 / 720]]
    #
    #     self.nn.forward(nn_input)
    #     output = self.nn.y
    #
    #     if output > 0.4:
    #         direction = 1
    #     else:
    #         direction = -1
    #
    #     return direction

    def think(self, mode, box_lists, agent_position, velocity):

        # x_box1 = x_box2 = CONFIG['WIDTH']
        # y_box1 = y_box2 = CONFIG['HEIGHT'] / 2
        x_box1 = x_box2 = 1280
        y_box1 = y_box2 = 360

        for box in box_lists:
            if box.x > agent_position[0]:
                x_box1 = agent_position[0] - box.x
                y_box1 = agent_position[1] - box.gap_mid
                if len(box_lists) > 1:
                    x_box2 = agent_position[0] - box_lists[box_lists.index(box) + 1].x
                    y_box2 = agent_position[1] - box_lists[box_lists.index(box) + 1].gap_mid
                else:
                    x_box2 = box.x + 200
                    y_box2 = y_box1
                break
            # else:
            #     # x_box1 = x_box2 = CONFIG['WIDTH']
            #     # y_box1 = y_box2 = CONFIG['HEIGHT'] / 2
            #     x_box1 = x_box2 = 1280
            #     y_box1 = y_box2 = 360
            #     break

        distance = math.sqrt((x_box1 ** 2) + (y_box1 ** 2))
        # max_distance = math.sqrt((CONFIG['WIDTH'] ** 2) + (CONFIG['HEIGHT'] ** 2))
        # max_distance = 1468.6

        # parameters = [[velocity / 10], [distance / max_distance], [x_box1 / CONFIG['WIDTH']],
        #               [y_box1 / CONFIG['HEIGHT']], [x_box2 / CONFIG['WIDTH']], [y_box2 / CONFIG['HEIGHT']]]

        parameters = [[velocity / 10], [distance / 1468.6], [x_box1 / 1280],
                      [y_box1 / 720], [x_box2 / 1280], [y_box2 / 720]]

        self.nn.forward(parameters)
        result = self.nn.y

        direction = -1

        # if mode == 'thrust' or CONFIG['seed'] > 1:
        #     if len(self.nn.y) == 3:
        #         maximum = max(result)
        #         if maximum == result[1]: direction = -0
        #         if maximum == result[2]: direction = 1
        #         return direction
        #     else:
        #         if result > 0.7:
        #             direction = 1
        #         elif result < 0.3:
        #             direction = -1
        #         else:
        #             direction = 0
        # else:  # helicopter or gravity
        #     if len(self.nn.y) == 2:
        #         maximum = max(result)
        #         if maximum == result[1]: direction = 1
        #         return direction
        #     else:
        if result > 0.4:
            direction = 1

        return direction

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided
