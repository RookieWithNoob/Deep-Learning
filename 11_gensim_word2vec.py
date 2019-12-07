from gensim.models import word2vec
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


raw_sentences = ["the Quick brown fix jumps over the iazy dogs", "yoyoyo you go home now to sleep"]

sentences = [s.split() for s in raw_sentences]

print(sentences)

model = word2vec.Word2Vec(sentences, min_count=1)

# 相似程度
print(model.similarity("go", "you"))
