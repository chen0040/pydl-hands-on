from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()

