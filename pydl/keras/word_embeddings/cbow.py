from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
from keras import backend as K

VOCAB_SIZE = 5000
EMBED_SIZE = 300
WINDOW_SIZE = 1

model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=WINDOW_SIZE*2,
                    embeddings_initializer='glorot_uniform'))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBED_SIZE, )))
model.add(Dense(units=VOCAB_SIZE, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

weights = model.layers[0].get_weights()[0]

