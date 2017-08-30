from theano import tensor as T
import theano
import theano.printing

a = T.scalar()

pow = a ** 2

g = theano.grad(pow, a)

print(theano.printing.debugprint(g))
print(theano.printing.debugprint(theano.function([a], g)))
