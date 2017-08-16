from __future__ import division, print_function
from keras.layers.core import Activation, Dense, SpatialDropout1D, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_DIR = '../../../data'


def parse_sentences(filename):
    sents = []
    word_freqs = collections.Counter()
    fin = open(filename, 'rt')
    for line in fin:
        words = line.strip().lower().split()
        for word in words:
            word_freqs[word] += 1
        sents.append(words)
    fin.close()
    return sents, word_freqs


def get_or_else(dictionary, key, default_value):
    try:
        return dictionary[key]
    except KeyError:
        return default_value


def generate_batch(s_sents, s_word2index, t_sents, t_word2index, batch_size,maxlen):
    while True:
        indices = np.random.permutation(np.arange(len(s_sents)))
        ss_sents = [s_sents[idx] for idx in indices]
        ts_sents = [t_sents[idx] for idx in indices]
        si_sents = [[get_or_else(s_word2index, word, s_word2index['UNK']) for word in sent] for sent in ss_sents]
        ti_sents = [[t_word2index[word] for word in sent] for sent in ts_sents]
        num_batches = len(s_sents) // batch_size
        for i in range(num_batches):
            s_batch = si_sents[i * batch_size: (i+1) * batch_size]
            t_batch = ti_sents[i * batch_size: (i+1) * batch_size]
            sp_batch = sequence.pad_sequences(s_batch, maxlen=maxlen)
            tp_batch = sequence.pad_sequences(t_batch, maxlen=maxlen)
            tpc_batch = np_utils.to_categorical(tp_batch.reshape(-1, 1), num_classes=len(t_word2index)).reshape(batch_size, -1, len(t_word2index))
            yield sp_batch, tpc_batch


def top_3_categorical_accuracy(ytrue, ypred):
    return top_k_categorical_accuracy(ytrue, ypred, k=3)


s_sents, s_wordfreqs = parse_sentences(os.path.join(DATA_DIR, 'reuters-sent.txt'))
t_sents, t_wordfreqs = parse_sentences(os.path.join(DATA_DIR, 'reuters-pos.txt'))
sent_lengths = np.array([len(sent) for sent in s_sents])

print("# records: {:d}".format(len(s_sents)))
print("# unique words: {:d}".format(len(s_wordfreqs)))
print("# unique POS tags: {:d}".format(len(t_wordfreqs)))
print("# words/sentence: min: {:d}, max: {:d}, mean: {:.3f}, median: {:.0f}"
      .format(np.min(sent_lengths), np.max(sent_lengths),
              np.mean(sent_lengths), np.median(sent_lengths)))


MAX_SEQLEN = 50
S_MAX_FEATURES = 50000
T_MAX_FEATURES = 45

EMBED_SIZE = 300
HIDDEN_SIZE = 100

BATCH_SIZE = 64

NUM_EPOCHS = 50
NUM_ITERATIONS = 20

s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 1
s_word2index = {x[0]: i+2 for i, x in enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index['PAD'] = 0
s_word2index['UNK'] = 1
s_index2word = { v: k for k, v in s_word2index.items()}

t_vocabsize = len(t_wordfreqs) + 1
t_word2index = { x[0]: i+1 for i, x in enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index['PAD'] = 0
t_index2word = { v: k for k, v in t_word2index.items() }

test_size = int(0.3 * len(s_sents))
s_sents_train, s_sents_test = s_sents[0:-test_size], s_sents[-test_size:]
t_sents_train, t_sents_test = t_sents[0:-test_size], t_sents[-test_size:]
train_gen = generate_batch(s_sents_train, s_word2index, t_sents_train, t_word2index, BATCH_SIZE, MAX_SEQLEN)
test_gen = generate_batch(s_sents_test, s_word2index, t_sents_test, t_word2index, BATCH_SIZE, MAX_SEQLEN)
print(len(s_sents_train), len(s_sents_test))

model = Sequential()
model.add(Embedding(input_dim=s_vocabsize, output_dim=EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(GRU(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_train_samples = len(s_sents_train) // BATCH_SIZE
num_test_samples = len(s_sents_test) // BATCH_SIZE

hist_acc, hist_val_acc, hist_loss, hist_val_loss = [], [], [], []
for i in range(NUM_ITERATIONS):
    history = model.fit_generator(train_gen, steps_per_epoch=num_train_samples, epochs=NUM_EPOCHS, validation_data=test_gen, validation_steps=num_test_samples)
    hist_acc.extend(history.history['acc'])
    hist_val_acc.extend(history.history['val_acc'])
    hist_loss.extend(history.history['loss'])
    hist_val_loss.extend(history.history['val_loss'])
    Xtest, Ytest = test_gen.next()
    Ytest_ = model.predict(Xtest)
    ytest = np.argmax(Ytest, axis=2)
    ytest_ = np.argmax(Ytest_, axis=2)
    print("=" * 80)
    print("Iteration #: {:d}".format(i + 1))
    print("-" * 80)
    for j in range(min(5, Ytest.shape[0])):
        sent_ids = Xtest[j]
        sent_words = [s_index2word[idx] for idx in sent_ids.tolist()]
        pos_labels = [t_index2word[idx] for idx in ytest[j].tolist()]
        pos_preds = [t_index2word[idx] for idx in ytest_[j].tolist()]
        triples = [x for x in zip(sent_words, pos_labels, pos_preds) if x[0] != "PAD"]
        print("label:     " + " ".join([x[0] + x[1].upper() for x in triples]))
        print("predicted: " + " ".join([x[0] + x[2].upper() for x in triples]))
        print("-" * 80)


# plot loss and accuracy
plt.subplot(211)
plt.title("Accuracy")
plt.plot(hist_acc, color="g", label="Train")
plt.plot(hist_val_acc, color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(hist_loss, color="g", label="Train")
plt.plot(hist_val_loss, color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()