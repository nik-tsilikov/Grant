from fuzzywuzzy import fuzz, StringMatcher
import pandas as pd


df = pd.read_excel('orfo_and_typos.L1_5.xlsx')
correct_words = df.CORRECT
mistake_words = df.MISTAKE

i = 0;
# for index, row in df.iterrows():
print(fuzz.ratio(correct_words[1], correct_words[1]))
print(fuzz.ratio(correct_words[1], mistake_words[1]))

# //    i=i+1
