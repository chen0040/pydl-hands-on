from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np

INPUT_FILE = '../../../../data/umich-sentiment-train.txt'
VOCAB_SIZE = 5000
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 20
VERBOSE = 1

np.random.seed(42)

counter = collections.Counter()
fin = open(INPUT_FILE, "rt", encoding="utf8")
maxlen = 0
for line in fin:
    _, sent = line.strip().split("\t")
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}

xs, ys = [], []
fin = open(INPUT_FILE, 'rt', encoding="utf8")
for line in fin:
    label, sent = line.strip().split('\t')
    ys.append(label)
    words = [w.lower() for w in nltk.word_tokenize(sent)]
    wids = [word2index[w] for w in words]
    xs.append(wids)
fin.close()
X = pad_sequences(xs, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

model = Sequential()
model.add(Embedding(input_dim=vocab_sz, output_dim=EMBED_SIZE, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=VERBOSE, validation_data=[Xtest, Ytest])

plt.subplot(211)
plt.title("accuracy")
plt.plot(history.history['acc'], color='r', label='train')
plt.plot(history.history['val_acc'], color='b', label='validation')
plt.legend(loc='best')

plt.title('loss')
plt.plot(history.history['loss'], color='r', label='train')
plt.plot(history.history['val_loss'], color='b', label='validation')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

score = model.evaluate(Xtest, Ytest, verbose=1)
print("Test score: {:3f}, accuracy: {:3f}".format(score[0], score[1]))
