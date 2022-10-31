from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


letters_dict = {
    'спам': [12, 82], 
    'не спам': [27, 116]
}
letters_data = pd.DataFrame(letters_dict, index=['писем', 'слов'])
letters_data


words = ['Prize', 'Offer', 'Access', 'Membership', 'Bill', 'Online', 'Refund', 'Bonus', 'Purchase', 'Money']
word_dict = {
    'слово': [x.lower() for x in words],
    'спам': [1, 4, 3, 0, 0, 9, 17, 5, 24, 19],
    'не спам': [3, 5, 10, 2, 10, 13, 33, 0, 13, 27]
}
words_data = pd.DataFrame(word_dict)
words_data


words_data[words_data['слово'] == 'prize']['спам'].item()
words_data_old = words_data.copy()

words_data['P(x|spam)'] = [
    (words_data[words_data['слово'] == name_word]['спам'].item() + 1) / (letters_data['спам'][1] + len(words_data['слово']))
    for name_word in words_data['слово']
]
words_data

words_data['P(x|nospam)'] = [
    (words_data[words_data['слово'] == name_word]['не спам'].item() + 1) / (letters_data['не спам'][1] + len(words_data['слово']))
    for name_word in words_data['слово']
]

words_data['Ln(Pspam)'] = [np.log(elem) for elem in words_data['P(x|spam)']]
words_data['Ln(Pnospam)'] = [np.log(elem) for elem in words_data['P(x|nospam)']]
words_data

P_spam = letters_data['спам'][0] / (letters_data['спам'][0] + letters_data['не спам'][0])
P_nospam = letters_data['не спам'][0] / (letters_data['спам'][0] + letters_data['не спам'][0])
print(f"P_spam = {P_spam}, P_nospam = {P_nospam}")

task_str = 'Million Online Access Cash Bill Offer Money'
task_str = task_str.lower()
task_words = task_str.split()
task_words

F_spam = np.log(P_spam) + np.sum([
    words_data[words_data['слово'] == elem]['Ln(Pspam)'].item()
    for elem in task_words
    if elem in words_data['слово'].values
])

F_nospam = np.log(P_nospam) + np.sum([
    words_data[words_data['слово'] == elem]['Ln(Pnospam)'].item()
    for elem in task_words
    if elem in words_data['слово'].values
])

P_class_Spam = 1.0 / (1 + np.exp(F_nospam - F_spam))

print(f"F_spam = {F_spam}, F_nospam = {F_nospam}, P_spam_letter = {P_class_Spam}")


## Normal version

words_data_alt = words_data.copy()

r = len(task_words) - len(set(words_data_alt['слово']) & set(task_words))

F_spam = np.log(P_spam) + np.sum([
    np.log( (words_data[words_data['слово'] == elem]['спам'].item() + 1) / (letters_data['спам'][1] + len(words_data['слово']) + r) )
    if elem in words_data['слово'].values
    else np.log(1.0 / (letters_data['спам'][1] + len(words_data['слово']) + r))
    for elem in task_words
])

F_nospam = np.log(P_nospam) + np.sum([
    np.log( (words_data[words_data['слово'] == elem]['не спам'].item() + 1) / (letters_data['не спам'][1] + len(words_data['слово']) + r) )
    if elem in words_data['слово'].values
    else np.log(1.0 / (letters_data['не спам'][1] + len(words_data['слово']) + r))
    for elem in task_words
])

P_class_Spam = 1.0 / (1 + np.exp(F_nospam - F_spam))

print(f"F_spam = {F_spam}, F_nospam = {F_nospam}, P_spam_letter = {P_class_Spam}")

