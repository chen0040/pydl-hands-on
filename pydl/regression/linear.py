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
    tf_X = None
    tf_Y = None
    tfTheta = None
    tfPredict = None

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

        self.tf_X = tf.placeholder('float', shape=(None, self.dimension), name='X')
        self.tf_Y = tf.placeholder('float', shape=(None, 1), name='Y')
        self.tfTheta = tf.Variable(np.random.randn(self.dimension, 1), dtype='float', name='theta')
        self.tfPredict = tf.matmul(self.tf_X, self.tfTheta)
        tfCost = tf.reduce_sum(tf.pow(self.tfPredict - self.tf_Y, 2)) / (2 * self.training_size)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(tfCost)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for iter in range(self.iterations):
                for (x, y) in zip(X, Y):
                    x = np.reshape(x, newshape=(1, self.dimension))
                    y = np.reshape(y, newshape=(1, 1))
                    session.run(optimizer, feed_dict={
                        self.tf_X: x,
                        self.tf_Y: y
                    })

                self.theta = session.run(self.tfTheta)

                if iter % self.display_every == 0:
                    self.cost = session.run(tfCost, feed_dict={
                        self.tf_X: X,
                        self.tf_Y: Y
                    })

                    print("iteration #:", "%04d" % (iter + 1), "cost=", "{:.9f}".format(self.cost),
                          "theta=", self.theta)

    def transform(self, Xprime):
        Xprime = np.hstack((np.ones(shape=(Xprime.shape[0], 1)), Xprime))
        init = tf.global_variables_initializer()
        predicted = None
        with tf.Session() as session:
            session.run(init)
            predicted = session.run(self.tfPredict, feed_dict={
                self.tf_X: Xprime,
                self.tfTheta: self.theta
            })
        return predicted

