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

inputs = [1, 0, 1] # inputs[weather, train, leave]
weights = [2, 2, 6] # weights[weather, train, leave]

def sigmoid(inputs, weights, bias):
    productAndSumOfInputsWeight = (inputs * weights).sum()
    print ('Product Array:', productAndSumOfInputsWeight)

    z = productAndSumOfInputsWeight + (-bias)

    sigmoid = 1 / 1 + math.exp(-z)
    return sigmoid


print('Am I going to Festival:', sigmoid(np.array(inputs), np.array(weights), 5))