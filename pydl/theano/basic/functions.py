from theano import tensor as T
import theano
import numpy as np

a = T.matrix()
ex = theano.function([a], [T.exp(a), T.log(a), a ** 2])

print(ex(np.random.randn(3, 3).astype(theano.config.floatX)))

w = theano.shared(1.0)

x = T.scalar('x')

mul = theano.function([x], updates=[(w, w * x)])

mul(4)

print(w.get_value())
