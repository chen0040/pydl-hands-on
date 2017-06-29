import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


def train_given_activation(activation):
    model = Sequential()
    model.add(Dense(1, input_dim=500))
    model.add(Activation(activation=activation))
    model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

    data = np.random.random((1000, 500))
    labels = np.random.randint(2, size=(1000, 1))
    score = model.evaluate(data, labels, verbose=0)
    print("Activation: ", activation)
    print("Before Training:", model.metrics_names, score)
    model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
    score = model.evaluate(data, labels, verbose=0)
    print("After Training:", model.metrics_names, score)

train_given_activation("relu")
train_given_activation("tanh")
train_given_activation("sigmoid")
train_given_activation("hard_sigmoid")
train_given_activation("linear")
