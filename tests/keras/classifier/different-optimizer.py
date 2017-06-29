import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


def train_given_optimiser(optimiser):
    model = Sequential()
    model.add(Dense(1, input_dim=500))
    model.add(Activation(activation='sigmoid'))
    model.compile(optimizer=optimiser, loss='binary_crossentropy', metrics=['accuracy'])
    data = np.random.random((1000, 500))
    labels = np.random.randint(2, size=(1000, 1))
    score = model.evaluate(data, labels, verbose=0)
    print("Optimiser: ", optimiser)
    print("Before Training:", model.metrics_names, score)
    model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
    score = model.evaluate(data, labels, verbose=0)
    print("After Training:", model.metrics_names, score)


train_given_optimiser("sgd")
train_given_optimiser("rmsprop")
train_given_optimiser("adagrad")
train_given_optimiser("adadelta")
train_given_optimiser("adam")
train_given_optimiser("adamax")
train_given_optimiser("nadam")

