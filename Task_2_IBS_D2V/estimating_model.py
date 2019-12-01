import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

model = Doc2Vec.load("IBS_dup_model")

records1_split = np.load("records1_split.npy", allow_pickle=True)
records2_split = np.load("records2_split.npy", allow_pickle=True)
correct_answers = np.load("correct_answers.npy", allow_pickle=True)

print('Our model result accuracy:')

i = 0
scores = []
for index in records1_split:
    score = model.wv.n_similarity(records1_split[i], records2_split[i])
    if score > 0.6:
        scores.append(1)
    else:
        scores.append(0)

    # print(records1_split[i])
    print("Pair ID: ", i, ". Score: ",  score)
    i = i + 1


accuracy = accuracy_score(correct_answers, scores) * 100

print(accuracy)
