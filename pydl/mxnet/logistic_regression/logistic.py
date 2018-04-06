import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt

def logistic(z):
    return 1.0 / (1 + nd.exp(-z))

x = nd.arange(-5, 5, .1)
y = logistic(x)

plt.plot(x.asnumpy(), y.asnumpy())
plt.show()

