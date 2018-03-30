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

    x = nd.ones(shape=(3, 3))
    y = nd.arange(3)
    print('x = ', x)
    print('y = ', y)
    print('x + y = ', x + y)
    y = y.reshape(shape=(3, 1))
    print('y = ', y)
    print('x + y = ', x + y)

    a = x.asnumpy()
    print(type(a))
    y = nd.array(a)
    print(y)

    z = nd.ones(shape=(3, 3), ctx=mx.gpu(0))
    print(z)

    x_gpu = x.copyto(mx.gpu(0))
    print(x_gpu + z)

    print(x_gpu.context)
    print(z.context)

    z = nd.ones(shape=(3, 3))
    print('id(z) = ', id(z))
    z2 = z.copyto(mx.gpu(0))
    print('id(z) = ', id(z2))
    z3 = z.as_in_context(mx.gpu(0))
    print('id(z) = ', id(z3))
    print(z)
    print(z3)



if __name__ == '__main__':
    main()
