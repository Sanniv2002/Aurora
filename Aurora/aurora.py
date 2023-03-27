'''
    ver. 1.0
    A Simple tiny ANN Library using which a simple neural network can be synthesised and trained.
    Includes only Gradient Descent Optimization.
    Not Designed for NLP Models
'''
import numpy as np
import math
import dill

# Layer Class


class Layer:
    def __init__(self, input_size, output_size):
        # Matrix of dimensions output perceptrons X input perceptrons
        self.weights = np.random.randn(output_size, input_size)
        # Matrix of dimensions output perceptrons X 1
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activations(Layer):  # Parent class, add new activation functions inheriting this one
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Tanh(Activations):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activations):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activations):
    def __init__(self):
        def relu(x):
            np.maximum(0, x)

        def relu_prime(x):
            return x > 0

        super().__init__(relu, relu_prime)


class ELU(Activations):
    def __init__(self):
        def elu(x, alpha):
            return alpha*(math.e**x-1) if x < 0 else x

        def elu_prime(x, alpha):
            return alpha*(math.e**x) if x < 0 else 1


class SoftMax(Activations):
    def __init__(self):
        def softmax(x):
            return math.exp(x)/np.sum(math.exp(x))


# Loss Functions
def mse(y_desired, y_predicted):
    return np.mean(np.power(y_desired-y_predicted, 2))


def mse_prime(y_desired, y_predicted):
    return 2 * (y_predicted-y_desired) / np.size(y_desired)


def binary_cross_entropy(y_desired, y_predicted):
    return np.mean(-y_desired * np.log(y_predicted) - (1 - y_desired) * np.log(1 - y_predicted))


def binary_cross_entropy_prime(y_desired, y_predicted):
    return ((1 - y_desired) / (1 - y_predicted) - y_desired / y_predicted) / np.size(y_desired)


def progress_bar(progress, total, error, accuracy):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100-int(percent))
    # print(f"\r|{bar}| {percent:.2f}%", end='\r')
    print('|', bar, "| ", "%.2f" % percent, "%", " • Epochs: ",
          progress, "/", total, ", Loss: %f, Accuracy: " % error, accuracy, "%", sep='', end='\r')


# Training
def train(network, loss, loss_prime, X, Y, batch_size, epochs, learning_rate):
    for e in range(epochs):
        error = 0
        nr_correct = 0
        for x, y in zip(X, Y):
            output = x
            # Forward Propagation
            for layer in network:
                output = layer.forward(output)

            error += loss(y, output)
            nr_correct += int(np.argmax(output) == np.argmax(y))
            grad = loss_prime(y, output)
            # Backward Propagation
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(X)
        accuracy = round((nr_correct / batch_size) * 100, 3)
        progress_bar(e+1, epochs, error, accuracy)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def save_model(network, filename):
    filename = filename + ".dat"
    dill.dump(network, open(filename, "wb"))


def deploy_model(filename):
    network = dill.load(open(filename, "rb"), dill.HIGHEST_PROTOCOL)
    return network
