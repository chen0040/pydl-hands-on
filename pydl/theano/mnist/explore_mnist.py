import pickle, gzip
import matplotlib.pyplot as plt
import numpy as np

with gzip.open('../../data/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

print(train_set[0].shape)
print(train_set[1].shape)

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.cmap'] = 'gray'

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_set[0][i].reshape((28, 28)))
    plt.axis('off')
    plt.title(train_set[1][i])

plt.show()

