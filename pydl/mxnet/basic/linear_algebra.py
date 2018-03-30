import mxnet as mx
from mxnet import nd


def main():
    x = nd.arange(20)
    A = x.reshape(shape=(5, 4))
    print(A)
    print('A[2, 3] = ', A[2, 3])
    print('A[2, :] = ', A[2, :])
    print('A[:, 3] = ', A[:, 3])
    print('A.T = ', A.T)

    X = nd.arange(24).reshape(shape=(2, 3, 4))
    print(X)

    u = nd.array([1, 2, 4, 8])
    v = nd.ones_like(u) * 2
    print('u + v = ', u + v)
    print('u - v = ', u - v)
    print('u * v = ', u * v)
    print('u / v = ', u / v)

    B = nd.ones_like(A) * 3
    print('B = ', B)
    print('A + B = ', A + B)
    print('A * B = ', A * B)

    a = 2
    x = nd.ones(3)
    y = nd.zeros(3)
    print(x.shape)
    print(y.shape)
    print((a * x).shape)
    print((a * x + y).shape)

    print(nd.sum(u))
    print(nd.sum(A))

    print(nd.mean(A))
    print(nd.sum(A) / A.size)

    print(nd.dot(u, v))
    print(nd.sum(u * v))

    print(nd.dot(A, u))

    A = nd.ones(shape=(3, 4))
    B = nd.ones(shape=(4, 5))
    print(nd.dot(A, B))

    # L2 norm
    print(nd.norm(u))
    # L1 norm
    print(nd.sum(nd.abs(u)))



if __name__ == '__main__':
    main()
