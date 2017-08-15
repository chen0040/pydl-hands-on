from gensim.models import KeyedVectors
from keras.layers.core import Dense,Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np

np.random.seed(42)

INPUT_FILE = '../../../../data/umich-sentiment-train.txt'
WORD2VEC_MODEL = "../../../../data/GoogleNews-vectors-negative300.bin.gz"
VOCAB_SIZE = 5000
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 10
VERBOSE = 1

counter = collections.Counter()
fin=open(INPUT_FILE, 'rt', encoding="utf8")
maxlen = 0
for line in fin:
    _, sent = line.strip().split('\t')
    words = [w.lower() for w in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid+1
vocab_sz = len(word2index) + 1
index2word = {v: k for k, v in word2index.items()}
index2word[0] = "_UNK_"

ws, ys = [], []
fin = open(INPUT_FILE, 'rt', encoding="utf8")
for line in fin:
    label, sent = line.strip().split('\t')
    ys.append(label)
    words = [w.lower() for w in nltk.word_tokenize(sent)]
    wids = [word2index[w] for w in words]
    ws.append(wids)
fin.close()
W = pad_sequences(ws, maxlen=maxlen)
Y = np_utils.to_categorical(ys)

# load pre-trained word2vec model
word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)

# transform X using word2vec lookup
X = np.zeros((W.shape[0], EMBED_SIZE))
for i in range(W.shape[0]):
    E = np.zeros((EMBED_SIZE, maxlen))
    words = [index2word[wid] for wid in W[i].tolist()]
    for j in range(maxlen):
        try:
            E[:, j] = word2vec[words[j]]
        except KeyError:
            pass
    X[i, :] = np.sum(E, axis=1)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(units=32, input_dim=EMBED_SIZE, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=[Xtest, Ytest])

# plot loss function
plt.subplot(211)
plt.title("accuracy")
plt.plot(history.history["acc"], color="r", label="train")
plt.plot(history.history["val_acc"], color="b", label="validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# evaluate model
score = model.evaluate(Xtest, Ytest, verbose=1)
print("Test score: {:.3f}, accuracy: {:.3f}".format(score[0], score[1]))





