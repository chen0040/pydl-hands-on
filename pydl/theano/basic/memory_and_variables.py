from theano import tensor as T
import numpy as np
import theano
import theano.printing

theano.config.floatX = 'float32'

a = T.matrix()
b = a.transfer('cpu')
print(b.eval({a: np.ones((2, 2)).astype(theano.config.floatX)}))

a = T.matrix('a')
b = a ** 2
sq = theano.function([a], b)

print(theano.printing.debugprint(sq))