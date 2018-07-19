import pandas as pd
import nltk
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize, word_tokenize, ngrams

def ngramize(texts, n):
    output=[]
    for text in texts:
        output += ngrams(text,n)
    return output

chunksize = 10 ** 6
raw = pd.read_csv(u'/Users/jif/Donors_choose/train.csv', nrows=100)
# print(raw.head())

X = raw.values[:,13]
y = raw.values[:,15]

# Set up X like train (with y).
doc_list = X.tolist()
response_list = y.tolist()
train = zip(doc_list,response_list)
text = []

for i in range(len(train)):
    text.append(word_tokenize(train[i][0]))

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))

# Eliminate stop words

freq_n1 = nltk.FreqDist(dictionary)
# freq.plot(cumulative=False)
# print(dictionary)
# print(text)

######################################################
# Frequency distribution for bigrams, n-grams.
######################################################
# This creates n-grams (n=3) for sentences that make up our textself.
# Each sentence is a description of why a donation is needed for a particular project.
d3 = ngramize(text,n=3)
freq_n3 = nltk.FreqDist(d3)
freq_n3.plot(cumulative=False)


# Predict things

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
# dictionary = set(word.lower() for passage in res for word in word_tokenize(passage[0]))
# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
# t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]







#
