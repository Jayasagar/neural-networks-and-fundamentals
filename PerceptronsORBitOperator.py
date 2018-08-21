
# Stimulate OR bit operator using Perceptron neural function
def ORBitwiseOperation(x1, x2, bias):
    weight = 1
    if x1*weight + x2*weight + bias <= 0:
        return 0
    return 1

# Bias = 0 and Weight = 1
print('ORBitwiseOperation: 0,0', ORBitwiseOperation(0, 0, 0))
print('ORBitwiseOperation: 0,1', ORBitwiseOperation(0, 1, 0))
print('ORBitwiseOperation: 1,0', ORBitwiseOperation(1, 0, 0))
print('ORBitwiseOperation: 1,1', ORBitwiseOperation(1, 1, 0))

