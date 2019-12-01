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

#df = pd.read_excel('task1_questions_small_corrected.xlsx')
df = pd.read_excel('dataset_for_doc2vec_simple_pairs.xlsx')

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

print('Our model result accuracy:')

i = 0
scores = []
for index in records1_split:
    score = model.wv.n_similarity(records1_split[i], records2_split[i])
    if (score > 0.6):
        scores.append(1)
    else:
        scores.append(0)
    i = i+1
    print(records1_split[i])
    print("Pair ID: ", i,". Score: ",  score)



accuracy = accuracy_score(df.is_duplicate, scores) * 100

print(accuracy)