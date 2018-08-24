import numpy as np
import math

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

print('Exponent of a large number', math.exp(-3439) )

'''
Requirment: Decide whether to attend festival at home town ot not

Criteria 1: Weather rainy or not
Criteria 2: Can I go by train?
Criteria 3: Will I get leave in office?

Bias: If leave has more bias value then, others not taking any priority
Leave  = 6

Bias : If All have equal priority, then bias will be less
'''

inputs = [0, 0, 1] # inputs[weather, train, leave]
weights = [2, 3, 5] # weights[weather, train, leave]

def sigmoid(inputs, weights, bias):
    productOfInputsWeight = (inputs * weights)
    print ('Product of Array:', productOfInputsWeight, ', Length: ', productOfInputsWeight.ndim)
    sum = productOfInputsWeight.sum()
    print('Sum of Array', sum)

    z = sum + (-bias)
    print('Perceptron output:', z)

    # Sigmoid value should not impact the netwowrk as it alwys greater than 1.
    sigmoid = 1 / 1 + math.exp(-z)
    return sigmoid


# print('Am I going to Festival:', sigmoid(np.array(inputs), np.array(weights), 5))


# Excercise 1: Sigmoid neurons simulating perceptrons
'''
Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, c>0. 
Show that the behaviour of the network doesn't change
'''
def simulateSigmoidAsPerceptron(inputs, weights, bias):

    outputs = []
    global sigmoid

    if inputs.ndim == 2:
        for index in range(len(inputs)):
            print('index:', index)
            input = inputs[index]
            weight = weights[index]
            sigmoidValue = sigmoid(inputs, weights, bias[index])
            print('sigmoid Output:', sigmoidValue)
            outputs.append(sigmoidValue)
        return outputs;

    # If number of dimensions is 1

    # Weight should positive number > 0
    return sigmoid(inputs, outputs, bias)

sigmoidSimulationInput = np.array([
    [[0, 2], [0, 3], [1, 5]],
    [[1, 3], [1, 3], [0, 5]],
    [[0, 4], [1, 2], [1, 4]]
])

print ('sigmoidSimulationInput Array Shape:', sigmoidSimulationInput.shape)

inputsList = sigmoidSimulationInput[:, :, 0]
weightsList = sigmoidSimulationInput[:, :, 1]
biasList = [4, 5, 6]

# print ('Get Inputs from sigmoidSimulationInput', sigmoidSimulationInput[:, :, 0])

sigmoidOutputs =  simulateSigmoidAsPerceptron(np.array(inputsList), np.array(weightsList), biasList)
print('simulateSigmoidAsPerceptron:', sigmoidOutputs)

pyplot.plot(sigmoidOwutputs, biasList)
pyplot.xlabel('Sigmoid output')
pyplot.ylabel('Bias value')
pyplot.show()