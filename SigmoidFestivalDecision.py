import numpy as np
import math

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
    productAndSumOfInputsWeight = (inputs * weights).sum()
    print ('Product Array:', productAndSumOfInputsWeight)

    z = productAndSumOfInputsWeight + (-bias)

    sigmoid = 1 / 1 + math.exp(-z)
    return sigmoid


# print('Am I going to Festival:', sigmoid(np.array(inputs), np.array(weights), 5))


# Excercise 1: Sigmoid neurons simulating perceptrons
'''
Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, c>0. 
Show that the behaviour of the network doesn't change
'''
def simulateSigmoidAsPerceptron(inputs, weights, bias):
    # Weight should positive number > 0
    productAndSumOfInputsWeight = (inputs * weights).sum()
    print ('Product Array:', productAndSumOfInputsWeight)

    z = productAndSumOfInputsWeight + (-bias)

    # Sigmoid value should not impact the netwowrk as it alwys greater than 1.
    sigmoid = 1 / 1 + math.exp(-z)
    return sigmoid

sigmoidSimulationInput = np.array([
    [[0, 2], [0, 3], [1, 5]],
    [[1, 3], [1, 3], [0, 5]],
    [[0, 4], [1, 2], [1, 4]]
])

print ('sigmoidSimulationInput Array Shape:', sigmoidSimulationInput.shape)
# print('simulateSigmoidAsPerceptron:', simulateSigmoidAsPerceptron(np.array(inputs), np.array(weights), 4))