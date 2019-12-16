from pyxdameraulevenshtein import damerau_levenshtein_distance
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_ndarray
import pandas as pd

# print(damerau_levenshtein_distance('smtih', 'smith'))
# print(damerau_levenshtein_distance('snapple', 'apple'))
# print(damerau_levenshtein_distance('testing', 'testtn'))
# print(damerau_levenshtein_distance('saturday', 'sunday'))
# print(damerau_levenshtein_distance('Saturday', 'saturday'))
# print(damerau_levenshtein_distance('orange', 'pumpkin'))
# print(damerau_levenshtein_distance('gifts', 'profit'))
# print(damerau_levenshtein_distance('Sjöstedt', 'Sjostedt'))

# Импорт словаря
dict_df = pd.read_fwf("pldf-win.txt", names=["Words"])
dict = set(dict_df["Words"])

