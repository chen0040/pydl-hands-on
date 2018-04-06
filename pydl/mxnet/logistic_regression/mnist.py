import mxnet as mx
from mxnet import nd, gluon
import numpy as np

def transform(data, label):
    return data.astype(np.float32), label.astype(np.float32)

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

image, label = mnist_train[0]
print(image.shape, label)