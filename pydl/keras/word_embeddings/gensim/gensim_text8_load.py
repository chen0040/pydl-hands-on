from gensim.models import word2vec
import os

OUTPUT_DATA_DIR = "../../../cache"

model = word2vec.Word2Vec.load(os.path.join(OUTPUT_DATA_DIR, "word2vec_gensim.bin"))

print(list(model.wv.vocab.keys())[0:10])
print(model["woman"])
print(model.most_similar("woman"))
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
print(model.similarity('girl', 'woman'))
