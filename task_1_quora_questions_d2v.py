# Import required libraries
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


# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument


# Import Data

#                 dtype = {'question1' : str, 'question2' : str})
df = pd.read_excel('./questions_corrected_2.xlsx')
questions1 = df.question1
questions2 = df.question2

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

questions1 = df.question1
questions2 = df.question2

# print(questions1)
# print(questions2)
# # Remove stop words
# def remove_stopwords(text):
#     nltk.download('stopwords')
#     stops = set(stopwords.words("english"))
#     words = [w for w in text.lower().split() if not w in stops]
#     final_text = " ".join(words)
#     return final_text
#
# # Special Characters
# def remove_specical_characters(review_text):
#     re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", review_text )
#     re.sub(r"\'s", " 's ", final_text )
#     re.sub(r"\'ve", " 've ", final_text )
#     re.sub(r"n\'t", " 't ", final_text )
#     re.sub(r"\'re", " 're ", final_text )
#     re.sub(r"\'d", " 'd ", final_text )
#     re.sub(r"\'ll", " 'll ", final_text )
#     re.sub(r",", " ", final_text )
#     re.sub(r"\.", " ", final_text )
#     re.sub(r"!", " ", final_text )
#     re.sub(r"\(", " ( ", final_text )
#     re.sub(r"\)", " ) ", final_text )
#     re.sub(r"\?", " ", final_text )
#     re.sub(r"\s{2,}", " ", final_text )
#     return review_text
#
labeled_questions=[]
questions1_split = []
questions2_split = []
i = 0;
for index, row in df.iterrows():
    labeled_questions.append(TaggedDocument(questions1[i].split(), df[df.index == i].qid1))
    labeled_questions.append(TaggedDocument(questions2[i].split(), df[df.index == i].qid2))
    questions1_split.append(questions1[i].split())
    questions2_split.append(questions2[i].split())
    i = i + 1;


np.save("labeled_questions.npy", labeled_questions)

# Model Learning

model = Doc2Vec(dm = 1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)
model.build_vocab(labeled_questions)

# Train the model with 20 epochs

for epoch in range(20):
    model.train(labeled_questions, epochs=model.epochs, total_examples=model.corpus_count)
    print("Epoch #{} is complete.".format(epoch + 1))

model.save('quora_model')

word = 'Washington'
print("Similarity to word:" + word)
print(model.wv.most_similar(word))

print ('Our model result accuracy:')

i = 0;
scores = []
for index in questions1_split:
    score = model.wv.n_similarity(questions1_split[i], questions2_split[i])
    if (score > 0.6):
        scores.append(1)
    else:
        scores.append(0)
    i = i+1;
    # print("Pair ID: ", i,". Score: ",  score)



accuracy = accuracy_score(df.is_duplicate, scores) * 100

print (accuracy)