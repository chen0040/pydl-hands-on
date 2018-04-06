import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt

def logistic(z):
    return 1.0 / (1 + nd.exp(-z))


data_ctx = mx.cpu()
model_ctx = mx.cpu()

with open('../data/adult/a1a.train') as f:
    train_raw = f.read()

with open('../data/adult/a1a.test') as f:
    test_raw = f.read()

def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = nd.zeros(shape=(num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros(shape=(num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(train_lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2])-1
            X[i, index] = 1
    return X, Y

Xtrain, Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)

print(nd.sum(Ytrain) / len(Ytrain))
print(nd.sum(Ytest) / len(Ytest))

batch_size = 64

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain), batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest), batch_size=batch_size, shuffle=True)

net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

trainer = gluon.Trainer(params=net.collect_params(), optimizer='sgd', optimizer_params={
    'learning_rate': 0.01
})

def log_loss(output, y):
    yhat = logistic(output)
    return -nd.nansum((1-y) * nd.log(1-yhat) + y * nd.log(yhat))


epochs = 30
loss_sequence = []
num_examples = len(Xtrain)

for epoch in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = log_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print('Epoch %s, loss: %s' % (epoch, cumulative_loss))
    loss_sequence.append(cumulative_loss)

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)

num_correct = 0.0
num_total = len(test_data)

for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    output = net(data)
    prediction = ((nd.sign(output)+1) / 2)
    num_correct += nd.sum(prediction == label)

print('Accuracy: %0.3f (%s/%s)' % (num_correct.asscalar()/ num_total, num_correct, num_examples))


