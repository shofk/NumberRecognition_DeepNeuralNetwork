import random
import numpy as np

class Network(object):

    """
    Initializes the Netwotk object
    'sizes' is a vector containing the number of nodes. The number of elements
    corresponds with the number of layers
    """
    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    """
    Return the output of the network if 'a' is input.
    """
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    """
    - Trains the neural network using mini-batch stochastic gradient descentself.
    - The 'training_data' is a list of tuples '(x, y)' representing the training
    inputs and the desired outputs.
    - The 'epochs' and 'mini_batch_size' are the number of epochs to traing for,
    and the size of the mini-batches to use when sampling.
    - The 'eta' is the learning rate, n.
    - If 'test_data' is provided then the networt
    will be evaluated against the test data after each epoch, and partial
    progress printed out. This is useful for tracking progress, but slows things
    down substantially.
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = sum(1 for _ in test_data)
            print(sum)
        n = sum(1 for _ in training_data)
        for j in range(epochs):
            # randomly shaffle the training_data
            list_training_data = list(training_data)
            random.shuffle(list_training_data)
            training_data = zip(list_training_data[0], list_training_data[1])
            # create mini-batches for training
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batchh in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data: print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else: print("Epoch {0} complete".format(j))

    """
    Takes a small batch of training set to improve the parameters
    """
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layers
        zs = [] # list to store all the z vectores, layer by layers
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l - 1].tranpose())

        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

### Miscellaneous functions
""" sigmoid function """
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

""" derivative of sigmoid function """
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
