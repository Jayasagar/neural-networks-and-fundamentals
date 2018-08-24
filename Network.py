import numpy as np

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