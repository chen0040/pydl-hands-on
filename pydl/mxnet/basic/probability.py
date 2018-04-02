import mxnet as mx
from mxnet import nd
import numpy as np

def main():
    probabilities = nd.ones(shape=(6, )) / 6
    print(nd.sample_multinomial(probabilities, shape=(10, )))
    print(nd.sample_multinomial(probabilities, shape=(5, 6)))

    rolls = nd.sample_multinomial(probabilities, shape=1000)
    totals = nd.zeros(6)
    counts = nd.zeros((6, 1000))
    for i, roll in enumerate(rolls):
        totals[int(roll.asscalar())] += 1
        counts[:, i] = totals

    print(totals / 1000)

    x = nd.arange(1000).reshape((1, 1000)) + 1
    estimates = counts / x
    print(estimates[:, 0])
    print(estimates[:, 1])
    print(estimates[:, 100])

    from matplotlib import pyplot as plt
    plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
    plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
    plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
    plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
    plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
    plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
    plt.axhline(y=0.16666, color='black', linestyle='dashed')
    plt.legend()
    plt.show()

    a = np.arange(20)
    print(a)
    print(a[:-1])


if __name__ == '__main__':
    main()