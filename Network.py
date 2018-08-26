import numpy as np
import random

class Network(object):

    # the list sizes contains the number of neurons in the respective layers.
    def __init__(self, sizes):
        self.num_layers =  len(sizes)
        self.sizes = sizes
        '''
        The biases and weights in the Network object are all initialized randomly, 
        using the Numpy np.random.randn function to generate Gaussian distributions with mean 0 and standard deviation 1.
        This random initialization gives our stochastic gradient descent algorithm a place to start from
        '''
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def feedForward(self, a):
        '''
        This method helps to calculate the output from a particular LAYER  based on input, weight and bias
        a is input
        '''
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SDG(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        Train the neural network using mini-batch stochastic gradient descent.
        The training_data is a list of tuples (x, y) representing the training inputs and the desired outputs.
        The other non-optional parameters are self-explanatory.
        If test_data is provided then the network will be evaluated against the test data after each epoch,
        and partial progress printed out.  This is useful for tracking progress, but slows things down substantially.
        '''

        if test_data:
            n_test = len(test_data)
        n = len(training_data) # 50k records MNSIT dataset
        for j in range(epochs):
            random.shuffle(training_data) # Shuffle the training data

            mini_batches =  [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batche in mini_batches:
                self.update_mini_batch(mini_batche, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)