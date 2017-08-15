from keras.layers import Merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

VOCAB_SIZE = 5000
EMBED_SIZE = 300

word_model = Sequential()
word_model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, embeddings_initializer='glorot_uniform',
                         input_length=1))
word_model.add(Reshape((EMBED_SIZE, )))

context_model = Sequential()
context_model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, embeddings_initializer='glorot_uniform',
                            input_length=1))
context_model.add(Reshape((EMBED_SIZE, )))

model = Sequential()
model.add(Merge([word_model, context_model], mode='dot'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')

merge_layer = model.layers[0]
word_model = merge_layer.layers[0]
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0]
