from pyaspeller import YandexSpeller
import pandas as pd
speller = YandexSpeller()
# text = 'В суббботу утромь.'
# changes = {change['word']: change['s'][0] for change in speller.spell(text)}
# for word, suggestion in changes.items():
#  text = text.replace(word, suggestion)
# print(text)

df = pd.read_excel('task4_data_for_univ_test_1.xlsx')
records = df.name_mtr

for index, row in df.iterrows():
 text = row['name_mtr']
 print("Text before: ")
 print(text)
 changes = {change['word']: change['s'][0] for change in speller.spell(text)}
 for word, suggestion in changes.items():
  text = text.replace(word, suggestion)
 row['name_mtr'] = text
 print("Text after: ")
 print(text)