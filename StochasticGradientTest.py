import Network as Network
# from keras.datasets import mnist
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

def main():
    print('Hello')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    net = Network([784, 30, 10])
    net.SGD(train_images, 30, 10, 3.0, test_data=test_images)


if __name__ == '__main__':
    main()