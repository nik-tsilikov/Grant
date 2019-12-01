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
#df = pd.read_excel('task1_questions_small_corrected.xlsx')
df = pd.read_excel('task1_questions_corrected.xlsx')

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

print("Data import completed")
# Remove stop words
nltk.download('stopwords')
stops = set(stopwords.words("english"))
def remove_stopwords(text):
    words = [w for w in text.lower().split() if not w in stops]
    final_text = " ".join(words)
    return final_text


# Special Characters
def remove_special_characters(review_text):
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", review_text )
    review_text = re.sub(r"\'s", " 's ", review_text )
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text )
    review_text = re.sub(r"\'re", " 're ", review_text )
    review_text = re.sub(r"\'d", " 'd ", review_text )
    review_text = re.sub(r"\'ll", " 'll ", review_text )
    review_text = re.sub(r",", " ", review_text )
    review_text = re.sub(r"\.", " ", review_text )
    review_text = re.sub(r"!", " ", review_text )
    review_text = re.sub(r"\(", " ( ", review_text )
    review_text = re.sub(r"\)", " ) ", review_text )
    review_text = re.sub(r"\?", " ", review_text )
    review_text = re.sub(r"\s{2,}", " ", review_text )
    return review_text

labeled_questions=[]
questions1_split = []
questions2_split = []

questions1 = df.question1
questions2 = df.question2
i = 0
for index, row in df.iterrows():
    question1 = questions1[i]
    question2 = questions2[i]
    question1 = remove_special_characters(question1)
    question2 = remove_special_characters(question2)
    question1 = remove_stopwords(question1)
    question2 = remove_stopwords(question2)
    if question1 != '' and question2 != '':
        labeled_questions.append(TaggedDocument(question1.split(), df[df.index == i].qid1))
        questions1_split.append(question1.split())
        labeled_questions.append(TaggedDocument(question2.split(), df[df.index == i].qid2))
        questions2_split.append(question2.split())
        print("Questions pair #" + str(i+1) + " of " + str(len(df.index)) + " labeled")
    else:
        print("Questions pair #" + str(i + 1) + " of " + str(len(df.index)) + " is empty and will not be processed")
    i = i + 1
print("Questions labeling completed")
#print(labeled_questions)
np.save("labeled_questions.npy", labeled_questions, allow_pickle=True)
np.save("questions1_split", questions1_split, allow_pickle=True)
np.save("questions2_split", questions2_split, allow_pickle=True)
print("Data saving completed")

# Model Learning

model = Doc2Vec(dm=1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)
model.build_vocab(labeled_questions)
print("Model building completed")

# Train the model with 20 epochs
for epoch in range(20):
    model.train(labeled_questions, epochs=model.epochs, total_examples=model.corpus_count)
    print("Epoch #{} is complete.".format(epoch + 1))

model.save('quora_model')
# print("Model learning completed")
# # word = 'Washington'
# # # word = 'speed'
# # print("Similarity to word:" + word)
# # print(model.wv.most_similar(word))

