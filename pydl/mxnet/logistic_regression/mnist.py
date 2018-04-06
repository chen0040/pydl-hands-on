import mxnet as mx
from mxnet import nd, gluon
import numpy as np

def transform(data, label):
    return data.astype(np.float32), label.astype(np.float32)

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

image, label = mnist_train[0]
print(image.shape, label)

num_inputs = 784
num_outputs = 10
num_examples = 60000

im = mx.nd.tile(image, (1, 1, 3))
print(im.shape)

import matplotlib.pyplot as plt

plt.imshow(im.asnumpy())
plt.show()