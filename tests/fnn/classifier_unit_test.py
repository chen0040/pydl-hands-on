import tensorflow as tf
import unittest
from tensorflow.examples.tutorials.mnist import input_data

from pydl.fnn.classifier import FnnClassifier


class FnnClassifierUnitTest(unittest.TestCase):

    def test_mnist(self):
        mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)
        classifier = FnnClassifier()
        # classifier.add_hidden_layer(10)
        classifier.fit(mnist.train)
        classifier.test(mnist.test.images,  mnist.test.labels)


if __name__ == '__main__':
    unittest.main()


