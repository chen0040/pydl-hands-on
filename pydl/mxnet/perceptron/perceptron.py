import mxnet as mx
from mxnet import nd, autograd
import matplotlib.pyplot as plt
import numpy as np

mx.random.seed(1)


def get_fake(samples, dimensions, epsilon):
    wfake = nd.random_normal(shape=(dimensions))
    bfake = nd.random_normal(shape=(1))
    wfake = wfake / nd.norm(wfake)

    X = nd.zeros(shape=(samples, dimensions))
    Y = nd.zeros(shape=(samples))

    i = 0
    while i < samples:
        tmp = nd.random_normal(shape=(1, dimensions))
        margin = nd.dot(tmp, wfake) + bfake
        if (nd.norm(tmp).asscalar() < 3) and (abs(margin.asscalar() > epsilon)):
            X[i, :] = tmp
            Y[i] = 1 if margin.ascalar() > 0 else -1
            i += 1
    return X, Y

def plot_data(X, Y):
    for (x, y) in zip(X, Y):
        if y.asscalar() == 1:
            plt.scatter(x[0].asscalar(), x[1].asscalar(), color='r')
        else:
            plt.scatter(x[0].asscalar(), x[1].asscalar(), color='b')
    plt.show()


def plot_score(w, d):
    xgrid = np.arange(-3, 3, 0.02)
    ygrid = np.arange(-3, 3, 0.02)
    xx, yy = np.meshgrid(xgrid, ygrid)
    zz = nd.zeros(shape=(xgrid.size, ygrid.size, 2))
    zz[:, :, 0]= nd.array(xx)
    zz[:, :, 1] = nd.array(yy)
    vv = nd.dot(zz, w) + d
    cs = plt.contour(xgrid, ygrid, vv.asnumpy())
    plt.clabel(cs, inline=1, fontsize=10)

X, Y = get_fake(50, 2, 0.3)
plot_data(X, Y)
plt.show()