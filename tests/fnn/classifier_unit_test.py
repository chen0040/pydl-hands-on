import tensorflow as tf
import unittest
from tensorflow.examples.tutorials.mnist import input_data

from pydl.fnn.classifier import FnnClassifier


class FnnClassifierUnitTest(unittest.TestCase):

    def test_mnist(self):
        mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)
        classifier = FnnClassifier()
        classifier.fit(lambda _: mnist.train.next_batch(100))


if __name__ == '__main__':
    unittest.main()


