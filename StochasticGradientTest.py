import Network
# from keras.datasets import mnist
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

def main():
    print('Hello')
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    net = Network.Network([784, 30, 10])
    net.SGD(train_images, 30, 10, 3.0, test_data=test_images)


if __name__ == '__main__':
    main()