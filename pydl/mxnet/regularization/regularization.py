from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
import numpy as np
mx.random.seed(1)

import matplotlib.pyplot as plt

mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(
    mnist['train_data'][:num_examples],
    mnist['train_label'][:num_examples].astype(np.float32)
), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(
    mnist['test_data'][:num_examples],
    mnist['test_label'][:num_examples].astype(np.float32)
), batch_size, shuffle=False)

num_inputs = 784
num_outputs = 10

data_ctx = mx.cpu()
model_ctx = mx.cpu()

W = nd.random.normal(shape=(num_inputs, num_outputs), ctx=data_ctx)
b = nd.random.normal(shape=num_outputs, ctx=data_ctx)

params = [W, b]

for param in params:
    param.attach_grad()


def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = nd.softmax(y_linear, axis=1)
    return yhat


def cross_entropy(yhat, y):
    return -nd.sum(y * nd.log(yhat), axis=0, exclude=True)

def penalty_l2(params):
    penalty = nd.zeros(shape=1)
    for param in params:
        penalty = penalty + nd.sum(param ** 2)
    return penalty


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def evaluate_accuracy(data_iterator, net):
    numerator = 0
    denominator = 0
    loss_avg = 0
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        loss = cross_entropy(output, label_one_hot)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)

    return (numerator / denominator).asscalar(), loss_avg


def plot_learning_curves(loss_tr, loss_ts, acc_tr, acc_ts):
    xs = list(range(len(loss_tr)))

    f = plt.figure(figsize=(12, 6))
    fig1 = f.add_subplot(121)
    fig2 = f.add_subplot(122)

    fig1.set_xlabel('epoch', fontsize=14)
    fig1.set_title('Comparing loss functions')
    fig1.semilogy(xs, loss_tr)
    fig1.semilogy(xs, loss_ts)
    fig1.grid(True, which='both')

    fig1.legend(['training loss', 'testing loss'], fontsize=14)

    fig1.set_xlabel('epoch', fontsize=14)
    fig2.set_title('Comparing accuracy')
    fig2.plot(xs, acc_tr)
    fig2.plot(xs, acc_ts)
    fig2.grid(True, which='both')

    fig2.legend(['training acc', 'testing acc'], fontsize=14)

    plt.show()


epochs = 1000
moving_loss = 0
niter = 0
l2_strength = .1

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot) + l2_strength * penalty_l2(params)
        loss.backward()
        SGD(params, 0.001)

        niter += 1
        moving_loss = 0.99 * moving_loss + .01 * nd.sum(loss).asscalar()
        est_loss = moving_loss / (1 - .99 * niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 100 == 99:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e + 1, train_loss, test_loss, train_accuracy, test_accuracy))


plot_learning_curves(loss_seq_train, loss_seq_test, acc_seq_train, acc_seq_test)
