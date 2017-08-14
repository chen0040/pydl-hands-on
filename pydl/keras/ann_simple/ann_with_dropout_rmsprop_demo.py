from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.datasets import mnist

RESHAPED = 784
NB_CLASSES = 10
OPTIMIZER = RMSprop()
EPOCHES = 20
N_HIDDEN = 128
BATCH_SIZE = 128
DROPOUT = 0.3
VERBOSE = 1
VALIDATION_SPLIT = 0.2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

x_train /= 255
x_test /= 255

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSE, epochs=EPOCHES, validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])



