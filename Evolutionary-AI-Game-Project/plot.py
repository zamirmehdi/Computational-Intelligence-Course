import csv

import numpy
from matplotlib import pyplot as plt


def input_data(path):
    nodes = []

    with open('data.csv', mode='r') as csv_file:
        data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for d in (data):
            nodes.append([float(d[0]), float(d[1]), float(d[2])])

    return nodes


nodes = input_data('data.csv')

print(nodes)
plt.plot(numpy.array(nodes))
plt.show()
