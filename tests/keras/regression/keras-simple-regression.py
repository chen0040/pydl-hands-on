import numpy as np
import math

from keras.layers import Dense
from keras.models import  Sequential

def normalize(array):
    return (array - array.mean()) / array.std()

np.random.seed(42)
num_house = 160
house_size = np.random.randint(low=1000, high=3500, size=(num_house, 1))
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=(num_house,1))

num_train_samples = math.floor(num_house * 0.7)

train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asarray(house_price[:num_train_samples])

test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

model = Sequential()
model.add(Dense(1, input_shape=(1, ), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(train_house_size_norm, train_house_price_norm,nb_epoch=300)

score = model.evaluate(test_house_size_norm, test_house_price_norm)

print("\nloss on test: {0}".format(score))