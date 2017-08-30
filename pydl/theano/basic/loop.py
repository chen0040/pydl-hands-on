from theano import tensor as T
import theano
import numpy as np

a = T.matrix()
b = T.matrix()


def fn(x):
    return x + 1


results, updates = theano.scan(fn, sequences=a)

f = theano.function([a], results, updates=updates)

print(f(np.ones((2, 3)).astype(theano.config.floatX)))

# print cumulative sum in a vector

a = T.vector()
s0 = T.scalar('s0')


def fn(current_element, prior):
    return prior + current_element


results, updates = theano.scan(fn=fn, outputs_info=s0, sequences=a)

f = theano.function([a, s0], results, updates=updates)

print(f([0, 3, 5], 0))

# print cumulative sum in a matrix

a = T.matrix()
s0 = T.scalar('s0')


def fn(current_element, prior):
    return prior + current_element.sum()


results, updates = theano.scan(fn=fn, sequences=a, outputs_info=s0)

f = theano.function([a, s0], results, updates=updates)

print(f(np.ones((20, 5)).astype(theano.config.floatX), 0))

# print cumulative sum in a vector using non_seq

a= T.vector()
inc = T.scalar('inc')


def fn(current_element, prior, step):
    return prior * step + current_element


results, updates = theano.scan(fn=fn, sequences=a, non_sequences=inc, outputs_info=T.constant(0.0, dtype=theano.config.floatX), n_steps=10)

f = theano.function([a, inc], results, updates=updates)

print(f(np.ones((20)).astype(theano.config.floatX), 5))