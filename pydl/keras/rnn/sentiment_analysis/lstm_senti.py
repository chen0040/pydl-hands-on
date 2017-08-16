from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

np.random.seed(42)

DATA_DIR = '../../../data'
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), 'rt', encoding='utf8')
for line in ftrain:
    label, sentence = line.strip().split('\t')
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v: k for k, v in word2index.items()}

X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open(os.path.join(DATA_DIR, 'umich-sentiment-train.txt'), 'rt', encoding='utf8')
for line in ftrain:
    label, sentence = line.strip().split('\t')
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index['UNK'])
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()
X = pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=[Xtest, ytest])

plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['acc'], color='g', label='Train')
plt.plot(history.history['val_acc'], color='b', label='Validation')
plt.legend(loc='best')

plt.subplot(212)
plt.title('Loss')
plt.plot(history.history['loss'], color='g', label='Train')
plt.plot(history.history['val_loss'], color='b', label='Validation')
plt.legend(loc='best')

plt.tight_layout()
plt.show()



