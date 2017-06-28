import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



image_rows = 28
image_cols = 28

mnist = input_data.read_data_sets("../../../MNIST_data/", one_hot=True)

train_images = mnist.train.images.reshape(mnist.train.images.shape[0],image_rows, image_cols, 1)
test_images =  mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

num_classes = 10
keep_prob = 0.5

input = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(input, nb_filter=32, filter_size=3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = conv_2d(network, nb_filter=64, filter_size=3, activation='relu', regularizer="L2")

network = max_pool_2d(network, 2)

network = fully_connected(network, 128, activation='tanh')

network = dropout(network, keep_prob)

network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.01,
                        loss='categorical_crossentropy', name='target')


num_epoch = 2
model = tflearn.DNN(network, tensorboard_verbose=0) 
model.fit({'input': train_images}, {'target': mnist.train.labels}, n_epoch=num_epoch,
           validation_set=({'input': test_images}, {'target': mnist.test.labels}),
            show_metric=True, run_id='TFLearn_DeepMNIST')