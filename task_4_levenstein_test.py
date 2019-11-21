import re
from fuzzywuzzy import fuzz
import pandas as pd


# Токенизация текста
def tokenize(text):
    return re.findall(r'[^\w\s]+|\w+', text)


# Сравнение каждого слова в строке с искомым
def find_variation(tokenized_string, standard):
    for word in tokenized_string:
        print(standart, word, fuzz.ratio(standard, word))
        if fuzz.ratio(standard, word) >= 60:
            return word


# Сравнение всех строк в датасете с искомым
def find_all_variations(standard, df):
    variations = []
    for index, row in df.iterrows():
        row['name_mtr'] = row['name_mtr'].replace('ё', 'е')
        tokenized = tokenize(row['name_mtr'])
        variation = find_variation(tokenized, standard)
        if variation:
            variations.append(variation)
    return set(variations)


def replace_words(s, words):
    for k, v in words.iteritems():
        s = s.replace(k, v)
    return s


# Удаление спецсимволов
def clean_text(review_text):
    review_text = ''.join(e for e in review_text if (e == ' ' or e.isalnum()))
    return ''.join(e for e in review_text if (e == ' ' or not e.isdigit()))


# Импорт словаря
dict_df = pd.read_fwf("pldf-win.txt", names=["Words"])
dict = set(dict_df["Words"])
print(dict)
print(len(dict))
# dict_df = dict_df.apply(lambda x: hash(tuple(x)), axis=1)
# dict_hash = dict_df[0:]
#
# # Импорт данных
# df = pd.read_excel('task4_data_for_univ_test_1.xlsx')
#
# for index, row in df.iterrows():
#     # Подготовка текста
#     row['name_mtr'] = clean_text(row['name_mtr'].replace('ё', 'е'))
#     tokenized = tokenize(row['name_mtr'])
#     temp = tokenized.copy()
#     for token in tokenized:
#         low_token = token.lower()
#         if (dict_df.loc[dict_df[0:] == low_token]).empty or len(token) == 1:
#             temp.remove(token)
#
#     tokenized = temp
#     print(tokenized)

# standart = row[0];
# print(standart)
# variations = set(find_all_variations(standart,df))
# #print(variations)
#
# variations_to_replace = dict.fromkeys("")
#
# for var in variations:
#   variations_to_replace[var] = standart
#
# #print(variations_to_replace)
# print("До замены")
# print(df['name_mtr'])
# print()
# d2 = {r'(\b){}(\b)'.format(k):r'\1{}\2'.format(v) for k,v in variations_to_replace.items()}
#
# df['name_mtr'] = df['name_mtr'].replace(d2, regex=True)
# print("После замены")
# print(df['name_mtr'])
