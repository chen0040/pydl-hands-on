import numpy as np
import math
import tflearn

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

input = tflearn.input_data(shape=[None], name="InputData")
linear = tflearn.layers.core.single_unit(input, activation='linear', name='Linear')

reg = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learing_rate=0.01, name='regression')

model = tflearn.DNN(reg)

model.fit(train_house_size_norm, train_house_price_norm, n_epoch=1000)

print("Training complete")

print(" Weights: W={0}, b={1}\n".format(model.get_weights(linear.W), model.get_weights(linear.b)))

print(" Accuracy {0}".format(model.evaluate(test_house_size_norm, test_house_price_norm)))

