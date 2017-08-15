from gensim.models import word2vec
import logging
import os

DATA_DIR = "../../../data"
OUTPUT_DATA_DIR = "../../../cache"

class Text8Sentences(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen

    def __iter__(self):
        with open(os.path.join(DATA_DIR, "text8"), "rt") as ftext:
            text = ftext.read().split(" ")
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words

logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

sentences = Text8Sentences(50)

# embedding size is 300 and we only chooses words which appear at least 30 times in the text8
model = word2vec.Word2Vec(sentences=sentences, size=300, min_count=30)

model.init_sims(replace=True)
model.save(os.path.join(OUTPUT_DATA_DIR, "word2vec_gensim.bin"))







