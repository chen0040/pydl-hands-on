from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000
batch_size = 4

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


X = nd.random.normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = 0.01 * nd.random.normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise


train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                   batch_size=batch_size, shuffle=True)

net = gluon.nn.Dense(units=1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

square_loss = gluon.loss.L2Loss()

trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd', optimizer_params={
    'learning_rate': 0.0001
})

epochs = 10
loss_sequences = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size=batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print('Epoch %s, loss: %s' % (e, cumulative_loss / num_examples))
    loss_sequences.append(cumulative_loss)

params = net.collect_params()
for param in params.values():
    print(param.name, param.data())

import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(8, 6))
plt.plot(loss_sequences)

plt.grid(True, which='both')
plt.xlabel('epoch', fontsize=14)
plt.ylabel('average loss', fontsize=14)
plt.show()

