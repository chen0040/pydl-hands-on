from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

np.random.seed(1671)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

RESHAPE = 784
NB_CLASSES = 10
N_HIDDEN = 128
BATCH_SIZE= 128
DROPOUT = 0.3
OPTIMIZER = SGD()
EPOCHES = 250
VALIDATION_SPLIT = 0.2
VERBOSE = 1

x_train = x_train.reshape(60000, RESHAPE)
x_test = x_test.reshape(10000, RESHAPE)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPE,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=EPOCHES, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)

score = model.evaluate(x_test, y_test)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])

