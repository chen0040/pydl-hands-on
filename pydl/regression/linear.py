import tensorflow as tf
import numpy as np


class LinearRegression(object):
    dimension = None
    learning_rate = None
    iterations = None
    display_every = 5
    theta = None
    cost = None
    training_size = None

    def fit(self, X, Y, learning_rate=None, iterations=None, display_every=None):
        self.dimension = X.shape[1] + 1
        self.training_size = X.shape[0]
        X = np.hstack((np.ones(shape=(self.training_size, 1)), X))

        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.2

        if iterations is not None:
            self.iterations = iterations
        else:
            self.iterations = 50

        if display_every is not None:
            self.display_every = display_every
        else:
            self.display_every = 5

        tfX = tf.placeholder('float', shape=(None, self.dimension), name='X')
        tfY = tf.placeholder('float', shape=(None, 1), name='Y')
        tfTheta = tf.Variable(np.random.randn(self.dimension, 1), dtype='float', name='theta')
        tfPredict = tf.matmul(tfX, tfTheta)
        tfCost = tf.reduce_sum(tf.pow(tfPredict - tfY, 2)) / (2 * self.training_size)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(tfCost)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for iter in range(self.iterations):
                for (x, y) in zip(X, Y):
                    x = np.reshape(x, newshape=(1, self.dimension))
                    y = np.reshape(y, newshape=(1, 1))
                    session.run(optimizer, feed_dict={
                        tfX: x,
                        tfY: y
                    })

                self.theta = session.run(tfTheta)

                if iter % self.display_every == 0:
                    self.cost = session.run(tfCost, feed_dict={
                        tfX: X,
                        tfY: Y
                    })

                    print("iteration #:", "%04d" % (iter + 1), "cost=", "{:.9f}".format(self.cost),
                          "theta=", self.theta)

    def transform(self, Xprime):
        Xprime = np.hstack()
