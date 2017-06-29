import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(32, input_dim=500))
model.add(Activation(activation='sigmoid'))
model.add(Dense(1))
model.add(Activation(activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_squared_error'])

data = np.random.random((1000, 500))
labels = np.random.randint(2, size=(1000, 1))

score = model.evaluate(data,labels, verbose=0)
print("Before Training:", model.metrics_names, score)

model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
score = model.evaluate(data,labels, verbose=0)
print("After Training:", model.metrics_names, score)
