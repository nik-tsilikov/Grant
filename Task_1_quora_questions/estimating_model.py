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

model = Doc2Vec.load("quora_model")

questions1_split = np.load("questions1_split.npy", allow_pickle=True)
questions2_split = np.load("questions2_split.npy", allow_pickle=True)

#df = pd.read_excel('task1_questions_small_corrected.xlsx')
df = pd.read_excel('task1_questions_corrected.xlsx')

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

print('Our model result accuracy:')

i = 0
scores = []
for index in questions1_split:
    score = model.wv.n_similarity(questions1_split[i], questions2_split[i])
    if (score > 0.6):
        scores.append(1)
    else:
        scores.append(0)
    i = i+1
    print(questions1_split[i])
    print("Pair ID: ", i,". Score: ",  score)



accuracy = accuracy_score(df.is_duplicate, scores) * 100

print (accuracy)