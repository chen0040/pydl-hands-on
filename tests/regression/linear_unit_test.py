import unittest
import numpy as np
import math

from pydl.regression.linear import LinearRegression


def normalize(array):
    return (array - array.mean()) / array.std()


class LinearRegressionUnitTest(unittest.TestCase):
    def test_simple(self):
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

        print(train_house_size_norm.shape)

        lr = LinearRegression()

        lr.fit(train_house_size_norm, train_house_price_norm)

        print('theta: ' + str(lr.theta))

        predicted = lr.transform(test_house_size_norm)

        print('predicted: ' + str(predicted))

if __name__ == '__main__':
    unittest.main()

