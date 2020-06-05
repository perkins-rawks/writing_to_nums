"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        # each layer is a column of the network
        self.num_layers = len(sizes)
        self.sizes = sizes

        # Each neuron has its own bias
        # 2nd layer = [[a],[b],[c]] for a, b, c in [0, 1] 
        # e.g. biases =   [ [[3],[2],[1]],  # first column
        #                           [[2]]   # second column]
        # in this example for sizes = [2, 3, 1]

        # randn(m,n) makes an m x n matrix of values from standard distribution
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # zip(sizes[:-1], sizes[1:]) creates a zip object (a pair)
        # sizes[:-1] -> [2_input, 3_hidden)]
        # sizes[1:] -> [3_hidden, 1_output]
        # Zip Links 2_input with 3_hidden and 3_hidden with 1_output
        # {(2_input, 3_hidden), (3_hidden, 1_output)} 
        # np.random.randn(3, 2) creates a 3 x 2 array, then a 1 by 3 array in 3
        # dimensions where each row is in a list.
        # Weights \in M_{3,2} e.g. -> [ [[1,3], [2,4], [5,3]] , [[1, 3, 4]] ]
        # The rows of the weights matrix represent the weights of every neuron
        # in receiving column.
        # The columns of the weights matrix represent the weights from each
        # neuron in the previous column.
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        # overall training data
        n = len(training_data)
        # xrange - ~same as range in Python 3
        for j in xrange(epochs):
            # Shuffle the batch, then split into different batches and update
            # them
            random.shuffle(training_data)
            # creates an list of size mini_batch_size containing    
            # for (x,y) in training_data
            # mini_batches =
            #  [ [(x,y)_0, (x,y)_1, ... (x,y)_mini_batch_size]_0,
            #    [(x,y)_0, (x,y)_1, ... (x,y)_mini_batch_size]_mini_batch_size,
            #                       ...
            #    [(x,y)_0, (x,y)_1, ... (x,y)_mini_batch_size]_n] ]
            mini_batches = [
                # k:k + mini_batch_size is a range from k to k + mini_batch_size
                training_data[k:k + mini_batch_size]
                # xrange (lower, upper, jump)
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # printing out every Epoch
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # vector.shape is the dimension of that vector
        # np.zeros(shape) creates a vector of the given shape full of zeros

        # nabla_b and nabla_w are the slight changes in biases and weights that 
        # we adjust by Calculus methods
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # does delta nabla imply 2nd derivatives?
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # nabla b = nabla b + delta nabla b for each item in the respective 
            # vectors
            # nabla w = nabla w + delta nabla w for each item in the respective 
            # vectors 
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # There is a negative because we want the opposite of the gradient
        # because that is the way to minimize the cost function (the steepest descent)  
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward

        # x is a handwritten image from the MNIST database
        # x is a vector of grey scale values in the file
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        # this runs once every connection
        for b, w in zip(self.biases, self.weights):
            # definition of z
            z = np.dot(w, activation) + b
            zs.append(z)
            # from book
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass

        # the error - BP1
        delta = self.cost_derivative(activations[-1], y)  *\
            sigmoid_prime(zs[-1])

        # dC/db BP3
        nabla_b[-1] = delta

        # dC/dw BP4
        # It's transpose here is a Python-specific implementation of the equation
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # BP2
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # BP3
            nabla_b[-l] = delta

            # BP4
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # alg returns [0.3, 0.4, 0, 0.999, 0, 0, 0, 0, 0, 0.5] -> recognizes a 3
        # expected is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] -> test is a handwritten 3
        # np.argmax(list): returns the index of the max value of the list
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
           \partial a for the output activations."""
        return (output_activations - y)

#### Miscellaneous functions
# z is a vector 
# sigmoid returns a vector with each value in it being sigmoid(elements of z)
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
