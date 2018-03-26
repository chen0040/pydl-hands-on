import mxnet as mx
from mxnet import nd


def main():
    mx.random.seed(1)
    x = nd.empty((3, 4))
    print(x)
    x = nd.zeros((3, 5))
    print(x)
    x = nd.ones((3, 4))
    print(x)
    y = nd.random.normal(0, 1, (3, 4))
    print(y)
    print(y.shape)
    print(y.size)
    print(x + y)
    print(x * y)
    print(nd.exp(y))
    print(nd.dot(x, y.T))

    print('id(y):', id(y))
    y = x + y
    print('id(y):', id(y))

    print('id(y):', id(y))
    y[:] = x + y
    print('id(y):', id(y))

    nd.elemwise_add(x, y, out=y)

    print('id(x):', x)
    x += y
    print('id(x):', x)

    print(x[1:3])

    print(x[1:2, 1:3])
    x[1:2, 1:3] = 5.0
    print(x)



if __name__ == '__main__':
    main()
