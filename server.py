import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Model

mnist = input_data.read_data_sets('/tmp/MNIST_data')
first_batch = mnist.train.next_batch(50)

m = Model()
encoding = [.5, .5]
print m.decode(encoding)

