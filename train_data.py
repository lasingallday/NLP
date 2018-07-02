import pandas as pd
import nltk
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize, word_tokenize

chunksize = 10 ** 6
raw = pd.read_csv(u'/Users/jif/Donors_choose/train.csv', nrows=100)
# print(raw.head())

X = raw.values[:,13]
y = raw.values[:,15]

# Set up X like train (with y).
doc_list = X.tolist()
response_list = y.tolist()
train = zip(doc,response)
text = []

for i in range(len(train)):
    text.append(word_tokenize(train[i][0]))

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
dictionary = (word.lower() for passage in res for word in word_tokenize(passage[0]))

# Eliminate stop words

freq = nltk.FreqDist(dictionary)
freq.plot(cumulative=False)


# Predict things

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
dictionary = set(word.lower() for passage in res for word in word_tokenize(passage[0]))
# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]







#
