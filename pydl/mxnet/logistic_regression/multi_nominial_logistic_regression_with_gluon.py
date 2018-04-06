from __future__ import print_function
import mxnet as mx
from mxnet import nd, gluon, autograd
import matplotlib.pyplot as plt
import numpy as np

data_ctx = mx.cpu()
model_ctx = mx.cpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000

def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)

train_raw = mx.gluon.data.vision.MNIST(train=True, transform=transform)
test_raw = mx.gluon.data.vision.MNIST(train=False, transform=transform)
train_data = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(train_raw), batch_size=batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(test_raw), batch_size=batch_size, shuffle=False)

net = gluon.nn.Dense(num_outputs)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), optimizer='sgd', optimizer_params={
    'learning_rate': 0.1
})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

print(evaluate_accuracy(test_data, net))

epochs = 10
moving_loss = 0

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss / num_examples, train_accuracy, test_accuracy))


def model_predict(net,data):
    output = net(data.as_in_context(model_ctx))
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                              10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    imtiles = nd.tile(im, (1,1,3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred=model_predict(net,data.reshape((-1,784)))
    print('model predictions are:', pred)
    break
