import theano
from theano import printing
import theano.tensor as T
import numpy as np

print(theano.config.device)
print(theano.config.floatX)
print(T.scalar())
print(T.iscalar())
print(T.fscalar())
print(T.dscalar())

x = T.matrix('x')
y = T.matrix('y')
z = x + y
print(z)
print(theano.pprint(z))
print(printing.debugprint(z))
print(theano.pp(z))
print(z.eval({x: [[1, 2], [1, 3]], y: [[1, 0], [3, 4]]}))
addition = theano.function([x, y], [z])
print(addition([[1, 2], [1, 3]], [[1, 0], [3, 4]]))
print(printing.debugprint(addition))
print(addition(np.ones((2, 2), dtype=theano.config.floatX), np.zeros((2, 2), dtype=theano.config.floatX)))

a = T.zeros((2, 3))
print(a.eval())
b = T.identity_like(a)
print(b.eval())
c = T.arange(10)
print(c.eval())
print(c.ndim)
print(c.dtype)
print(c.type)

a = T.matrix()
print(a.shape)

shape_func = theano.function([a], a.shape)

print(shape_func([[1, 2], [1, 3]]))

n = T.iscalar()
c = T.arange(n)
print(c.shape.eval({n: 10}))
print(c.eval({n:10}).shape)

a = T.arange(10)
b = T.reshape(a, (5, 2))
print(b.eval())

print(T.arange(10).reshape((5, 2))[::-1].T.eval())

a, b = T.matrices('a', 'b')

z = a * b

print(z.eval({a: np.ones((2, 2)).astype(theano.config.floatX), b: np.diag((3, 3)).astype(theano.config.floatX)}))

z = T.mul(a, b)

print(z.eval({a: np.ones((2, 2)).astype(theano.config.floatX), b: np.diag((3, 3)).astype(theano.config.floatX)}))

a = T.matrix()
b = T.scalar()

z = a * b
print(z.eval({a: np.diag((3, 3)).astype(theano.config.floatX), b : 3}))

cond = T.vector('cond')

x, y = T.vectors('x', 'y')

z = T.switch(cond, x, y)

print(z.eval({ cond: [1, 0], x: [10, 10], y: [3, 2]}))

a = T.matrix('a')

print(T.max(a).eval({a: [[1, 2], [3, 4]]}))

print(T.max(a, axis=0).eval({a: [[1, 2], [3, 4]]}))

print(T.max(a, axis=1).eval({a: [[1, 2], [3, 4]]}))

a = T.arange(10).reshape((5, 2))
b = a[::-1]

print(b.eval())
print(T.concatenate([a, b]).eval())
print(T.concatenate([a, b], axis=1).eval())
print(T.stack([a, b]).eval())

a = T.arange(10).reshape((5, 2))
print(T.set_subtensor(a[3:], [-1, -1]).eval())
print(T.inc_subtensor(a[3:], [-1, -1]).eval())

