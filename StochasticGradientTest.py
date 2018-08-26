import Network as Network
from keras.datasets import mnist

def main():
    print('Hello')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    net = Network([784, 30, 10])
    net.SGD(x_train, 30, 10, 3.0, test_data=x_test)


if __name__ == '__main__':
    main()