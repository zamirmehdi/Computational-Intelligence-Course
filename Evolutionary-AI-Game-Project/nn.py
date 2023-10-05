import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):
        # layer_sizes example: [4, 10, 2]

        self.w1 = np.random.normal(size=(layer_sizes[1], layer_sizes[0]))  # random.randn for normal or random.normal
        self.w2 = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))
        self.b1 = np.random.normal(size=(layer_sizes[1], 1))
        self.b2 = np.random.normal(size=(layer_sizes[2], 1))

        self.y = 0

    def activation(self, x):
        # return np.maximum(0, x)  # ReLU
        return 1 / (1 + np.exp(-x))  # Sigmoid

    def forward(self, x):
        # # x example: np.array([[0.1], [0.2], [0.3]])
        z1 = (self.w1 @ x) + self.b1
        out1 = self.activation(z1)

        z2 = self.w2 @ out1 + self.b2
        self.y = self.activation(z2)
