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
df = pd.read_excel('dataset_for_doc2vec_simple_pairs_with_typos_1.xlsx')

# Check for null values
df[df.isnull().any(axis=1)]

# Drop rows with null Values
df.drop(df[df.isnull().any(axis=1)].index, inplace=True)

print("Data import completed")
# Remove stop words
nltk.download('stopwords')
stops = set(stopwords.words("russian"))
def remove_stopwords(text):
    words = [w for w in text.lower().split() if not w in stops]
    final_text = " ".join(words)
    return final_text


# Special Characters
def remove_special_characters(review_text):
    review_text = re.sub(r"[^А-Яа-я0-9(),!.?\'`]", " ", review_text)
    # review_text = re.sub(r"\'s", " 's ", review_text )
    # review_text = re.sub(r"\'ve", " 've ", review_text)
    # review_text = re.sub(r"n\'t", " 't ", review_text )
    # review_text = re.sub(r"\'re", " 're ", review_text )
    # review_text = re.sub(r"\'d", " 'd ", review_text )
    # review_text = re.sub(r"\'ll", " 'll ", review_text )
    # review_text = re.sub(r",", " ", review_text )
    # review_text = re.sub(r"\.", " ", review_text )
    # review_text = re.sub(r"!", " ", review_text )
    # review_text = re.sub(r"\(", " ( ", review_text )
    # review_text = re.sub(r"\)", " ) ", review_text )
    # review_text = re.sub(r"\?", " ", review_text )
    # review_text = re.sub(r"\s{2,}", " ", review_text )
    return review_text

labeled_records = []
records1_split = []
records2_split = []
correct_answers = []

records1 = df.record1
records2 = df.record2
i = 0
for index, row in df.iterrows():
    record1 = records1[i]
    record2 = records2[i]
    record1 = remove_special_characters(record1)
    record2 = remove_special_characters(record2)
    record1 = remove_stopwords(record1)
    record2 = remove_stopwords(record2)
    if record1 != '' and record2 != '':
        labeled_records.append(TaggedDocument(record1.split(), df[df.index == i].rid1))
        records1_split.append(record1.split())
        labeled_records.append(TaggedDocument(record2.split(), df[df.index == i].rid2))
        records2_split.append(record2.split())
        correct_answers.append(row["is_duplicate"])
        print("Records pair #" + str(i+1) + " of " + str(len(df.index)) + " labeled")
    else:
        print("Records pair #" + str(i + 1) + " of " + str(len(df.index)) + " is empty and will not be processed")
    i = i + 1
print("Records labeling completed")

np.save("labeled_records.npy", labeled_records, allow_pickle=True)
np.save("records1_split", records1_split, allow_pickle=True)
np.save("records2_split", records2_split, allow_pickle=True)
np.save("correct_answers", correct_answers, allow_pickle=True)
print("Data saving completed")

# Model Learning

model = Doc2Vec(dm=1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)
model.build_vocab(labeled_records)
print("Model building completed")

# Train the model with 20 epochs
for epoch in range(20):
    model.train(labeled_records, epochs=model.epochs, total_examples=model.corpus_count)
    print("Epoch #{} is complete.".format(epoch + 1))

model.save('IBS_dup_model_with_typos')
# print("Model learning completed")
# # word = 'Washington'
# # # word = 'speed'
# # print("Similarity to word:" + word)
# # print(model.wv.most_similar(word))

