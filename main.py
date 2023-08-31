# This is a sample Python script.
import random
import numpy as np
import math
from keras.datasets import mnist

LAYER_SIZES = [784, 16, 10]
BATCHES = 1500
BATCH_SIZE = 32


def sigmoid(input):
    output = []
    for x in input:
        output.append(1 / (1 + math.e ** (-x)))
    return output


def d_sigmoid(input):
    k = [1 for x in input]
    return np.multiply(sigmoid(input), np.subtract(k, sigmoid(input)))


def linear(x):
    return x


def d_linear(x):
    return [1 for _ in range(len(x))]


def relu(input):
    output = []
    for x in input:
        if x <= 0:
            output.append(0)
        else:
            output.append(x)
    return output


def d_relu(input):
    output = []
    for x in input:
        if x <= 0:
            output.append(0)
        else:
            output.append(1)
    return output


"""
Finds the most likely number from a list of probabilities
@param output: the output layer

@return max: the most likely output
"""


def select_ans(output):
    max_val = 0
    for i in range(len(output)):
        if output[i] > output[max_val]:
            max_val = i
    return max_val


"""
turns x in the input array to [0, 1]
"""


def norm(x_vals, max_value):
    return np.multiply(1 / max_value, np.ndarray.flatten(x_vals))


class RNN:
    ACT_FX = {"relu": [relu, d_relu], "linear": [linear, d_linear], "sigmoid": [sigmoid, d_sigmoid]}

    def __init__(self, layers, max_val):
        self.max_val = max_val
        self.acts = []
        self.layer_sizes = []
        self.layers = []
        for k in range(len(layers[0:-1])):
            self.layers.append([[random.uniform(-0.005, 0.005) for _ in range(layers[k + 1]["size"])] for _ in
                                range(layers[k]["size"] + 1)])
            if not RNN.ACT_FX.get(layers[k]["act"]):
                raise ValueError("No such function as: {act}".format(act=layers[k]["act"]))
            self.acts.append(RNN.ACT_FX[layers[k]["act"]])
            self.layer_sizes.append(layers[k]["size"])
        self.layer_sizes.append(layers[-1]["size"])

    def load(self, files):
        self.layers = [np.loadtxt(open(x), delimiter=',') for x in files]

    """
    turns the weights and biases in a file to random floats in (-1, 1)
    """

    def reset(self):
        for i in range(len(self.layer_files)):
            a = [[random.uniform(-0.005, 0.005) for _ in range(self.layer_sizes[i + 1])] for
                 _ in range(self.layer_sizes[i] + 1)]
            np.savetxt(self.layer_files[i], a, delimiter=',')

    """
    returns the value of the nodes of each layer except input layer
    @param x_vals: x_train values
    @param layers: matrix of weights and biases
    @param act: activation functions array
    @return: array with neurons of each layer
    """

    def forward_pass(self, x_vals, layers):
        output = [self.acts[0][0](np.add(np.matmul(x_vals, layers[0][0:-1]), layers[0][-1]))]

        for i in range(len(layers) - 1):
            output.append(self.acts[i + 1][0](np.add(np.matmul(output[i], layers[i + 1][0:-1]), layers[i + 1][-1])))

        return output

    """
    Applies gradient descent to layers of nn on 1 training example
    @param x_vals: input value for training data
    @param output_value: class of training data
    @param layers: array of matrices that represent weights and bias
    @param act: array of activation function per layer
    @param d_act: derivative of those functions
    @param l_rate: learn rate
    @return: new layers
    """

    def back_prop(self, x_vals, output_value, layers, l_rate):
        new_layers = []

        nodes = self.forward_pass(x_vals, layers)  # value of neurons for hidden/output layers
        # calculate partial derivative of cost based on forward prop and outputs
        output = [0 for _ in range(LAYER_SIZES[-1])]
        output[output_value] = 1

        # calculate the pd of cost by z2 for output layer
        zout = np.add(np.matmul(nodes[0], layers[1][0:-1]), layers[1][-1])
        dc_dzout = np.multiply(np.subtract(nodes[-1], output), self.acts[1][1](zout))  # ∂cost/∂z2

        # gradient descent on output layer
        dc_dw2 = np.matmul(np.array([dc_dzout]).T, [nodes[0]]).T  # ∂cost/∂w2
        d_layerout = np.concatenate((dc_dw2, [dc_dzout]))  # add bias to end of weight file
        grad_layerout = np.multiply(l_rate, d_layerout)
        new_layers.append(np.subtract(layers[1], grad_layerout))  # set new layer to old layer with gradient descent

        # calculate gradient for hidden layers
        for i in reversed(range(len(layers) - 1)):
            # calculate pd of cost by z1 for layer1 (hidden layer)
            zj = nodes[i]
            dc_dzj = np.multiply(np.dot(layers[i + 1][0: -1], dc_dzout), self.acts[i][1](zj))  # ∂cost/∂z1

            #  gradient descent on layer1
            dc_dwj = np.matmul(np.array([dc_dzj]).T, [x_vals]).T  # ∂cost/∂w1
            d_layerj = np.concatenate((dc_dwj, [dc_dzj]))  # add bias to end of weight file
            grad_layerj = np.multiply(l_rate, d_layerj)
            new_layers.insert(0, np.subtract(layers[i], grad_layerj))  # set new layer to old layer with grad descent

        return new_layers

    def train(self, x_data, y_data, num_batches, batch_size, l_rate=1):
        for i in range(num_batches):
            # get average of batch
            batch = [self.back_prop(norm(x_data[x + i * batch_size], self.max_val), y_data[x + i * batch_size],
                                    self.layers, l_rate) for x in range(BATCH_SIZE) if
                     (x + i * batch_size) < len(x_data)]
            self.layers = [[1 / min(batch_size, len(batch)) * j for j in subarray] for subarray in
                           self.sum_arrays(batch)]

    def save(self, files):
        # save average to files
        for x in range(len(self.layers)):
            np.savetxt(files[x], self.layers[x], delimiter=',')

    def sum_arrays(self, data):
        sum = []
        for x in range(len(self.layer_sizes[0:-1])):
            sum.append(np.zeros([self.layer_sizes[x] + 1, self.layer_sizes[x + 1]]))
        for x in data:
            for p in range(len(data[0])):
                sum[p] = np.add(sum[p], x[p])
        return sum

    def predict(self, x_data):
        y_vals = [select_ans(self.forward_pass(norm(x, self.max_val), self.layers)[-1]) for x in x_data]
        return y_vals


if __name__ == '__main__':
    """Code for init"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_layers = [{"size": 784, "act": "relu"}, {"size": 16, "act": "sigmoid"}, {"size": 10}]
    rnn = RNN(n_layers, 255)
    rnn.train(x_train[:20], y_train[:20], 1500, 32, 0.5)

    total = 0
    for _ in range(2000):
        print("Expected val: " + str(y_test[_]) + " | Actual val: " + str(rnn.predict([x_test[_]])))
        if y_test[_] == rnn.predict([x_test[_]]):
            total += 1
    print("total: " + str(total))
