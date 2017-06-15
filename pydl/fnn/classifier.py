import tensorflow as tf
import numpy as np


class FnnClassifier(object):
    hidden_layers = None
    input_dimension = None
    output_dimension = None
    weights = None
    biases = None
    W_output = None
    b_output = None
    tf_X = None
    tf_Y = None
    predicted = None
    iterations = 1000
    learning_rate = 0.5
    cost = None
    bp = None
    session = None

    def __init__(self):
        self.hidden_layers = []
        self.weights = []
        self.biases = []

    def add_hidden_layer(self, size):
        self.hidden_layers.append(size)

    def build(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.tf_X = tf.placeholder(dtype=tf.float32, shape=(None, self.input_dimension))
        self.tf_Y = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension))
        input_dimension = self.input_dimension
        output = self.tf_X
        for i in range(len(self.hidden_layers)):
            input = output
            output_dimension = self.hidden_layers[i]
            W = tf.Variable(np.random.randn(input_dimension, output_dimension), dtype=tf.float32, name='W_' + str(i))
            self.weights.append(W)
            b = tf.Variable(np.random.randn(1, output_dimension), dtype=tf.float32, name='b_' + str(i))
            self.biases.append(b)
            output = tf.nn.relu(tf.matmul(input, W) + b)
            input_dimension = output_dimension

        self.W_output = tf.Variable(tf.zeros(shape=(input_dimension, self.output_dimension)), dtype=tf.float32, name='W_output')
        self.b_output = tf.Variable(tf.zeros(shape=(1, self.output_dimension)), dtype=tf.float32, name='b_output')
        self.predicted = tf.nn.softmax(tf.matmul(output, self.W_output) + self.b_output)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_Y, logits=self.predicted))

        self.bp = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def fit(self, batched_data):

        if self.session is not None:
            self.session.close()
            self.session = None

        X, Y = batched_data.next_batch(100)

        input_dimension = X.shape[1]
        output_dimension = Y.shape[1]
        self.build(input_dimension, output_dimension)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

        for iter in range(self.iterations):

            self.session.run(self.bp, feed_dict={
                self.tf_X: X,
                self.tf_Y: Y
            })
            cost = self.session.run(self.cost, feed_dict={
                self.tf_X: X,
                self.tf_Y: Y
            })
            # print('cost: ' + str(cost))

            X, Y = batched_data.next_batch(100)

    def test(self, test_X, test_Y):
        correct_prediction = tf.equal(tf.arg_max(self.tf_Y, 1), tf.arg_max(self.predicted, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        fd = {
            self.tf_X: test_X,
            self.tf_Y: test_Y
        }
        test_accuracy = self.session.run(accuracy, feed_dict=fd)
        print("Test accuracy: {0}%".format(test_accuracy * 100.0))