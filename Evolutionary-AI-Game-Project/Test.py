import csv
import random

import numpy as np

parameters = [[0], [1], [3], [2]]
np_list = np.array(parameters)
out = np_list / np.linalg.norm(np_list)
out = np_list / max(parameters)
print(type(np_list[0]))

print(out)

x = np.random.rand()
print(x)
x = np.random.rand()
print(x)


# parameters = [0, 1, 3, 2]
# print(random.sample(population=parameters, k=2))
#
#
# with open('data.csv', mode='w') as csv_file:
#     csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_writer.writerow(parameters)
#     csv_writer.writerow(parameters)

def think(self, mode, box_lists, agent_position, velocity):
    i = 0
    while mode == 'helicopter':

        if len(box_lists):
            if box_lists[i].x > agent_position[0]:

                x0 = agent_position[0] - box_lists[i].x
                y0 = agent_position[1] - box_lists[i].gap_mid

                if len(box_lists) > 1:
                    x1 = agent_position[0] - box_lists[i + 1].x
                    y1 = agent_position[1] - box_lists[i + 1].gap_mid
                else:
                    x1 = box_lists[i].x + 200
                    y1 = y0
                break
            else:
                i += 1
        else:
            x0 = x1 = 1280
            y0 = y1 = 360
            break
    z1 = math.sqrt((x0 ** 2) + (y0 ** 2))

    # nn_input = [[velocity / 10], [z1 / 1468.6],  [x0 / 1280],
    #             [y0 / 720], [x1 / 1280], [y1 / 720]] best result till now

    nn_input = [[velocity / 10], [z1 / 1468.6], [x0 / 1280],
                [y0 / 720], [x1 / 1280], [y1 / 720]]

    output = self.nn.forward(nn_input)[0]

    if output[0] > 0.4:
        direction = 1
    else:
        direction = -1

    return direction

# def think(self, mode, box_lists, agent_position, velocity):
#
#     # TODO
#     # mode example: 'helicopter'
#     # box_lists: an array of `BoxList` objects
#     # agent_position example: [600, 250]
#     # velocity example: 7
#
#     direction = -1
#
#     layer_sizes = self.init_network(mode)
#
#     # velocity * 100 to increase its effect
#     parameters = [[velocity * 10], [agent_position[0]], [agent_position[1]]]
#     x_distance = agent_position[0]
#
#     if box_lists:
#         # for box_list in box_lists:
#         #     parameters.append(box_list.x)
#         #     parameters.append(box_list.gam_mid)
#
#         parameters.append([box_lists[0].x])
#         parameters.append([box_lists[0].gap_mid])
#
#         x_distance = box_lists[0].x - agent_position[0]
#         y_distance = box_lists[0].gap_mid - agent_position[1]
#
#         parameters.append([x_distance])
#         parameters.append([y_distance])
#
#     # parameters.append([CONFIG['WIDTH']/10])
#     # parameters.append([CONFIG['HEIGHT']/10])
#
#     np_list = np.array(parameters)
#
#     # normalized_list = np_list/np.linalg.norm(np_list)
#     normalized_list = np_list / max(parameters)
#
#     self.nn.forward(normalized_list)
#     # print(self.nn.y)
#
#     if self.nn.y > 0.5:
#         direction = 1
#
#     return direction
