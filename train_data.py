import pandas as pd
import nltk
import NLTKPreprocessor
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize, word_tokenize
import build_model
from operator import itemgetter

chunksize = 10 ** 6

raw = pd.read_csv(u'/Users/jif/Donors_choose/train.csv', nrows=100)

# X = token_dict.values()
# print(raw.head())
X = raw.values[:,13]
y = raw.values[:,15]

# Use g for all rows of X.
# Do not need h.
document = X.tolist()
#h = ','.join(str(v) for v in g) # integers become strings here
# print(g)

# from nltk.corpus import movie_reviews as reviews
#
# X = [reviews.raw(fileid) for fileid in reviews.fileids()]
# y = [reviews.categories(fileid)[0] for fileid in reviews.fileids()]

# Find the common words that cause donations to occur.
# Add vectorizer.
document = document[0:5]
for i in range(len(document)):
    # print(document[i]+'\n')
    print(word_tokenize(document[i]))

# NLTKPreprocessor class works as a function, like so.
# preproc = NLTKPreprocessor.NLTKPreprocessor()
# res = preproc.tokenize(g)
# print(res)


# Set up X like train (with y).
doc = X.tolist()
is_approved = y.tolist()
doc = [doc[0]]
is_approved = [is_approved[0]]
text = []
# word_tokenize doesn't work. use split function.
for i in range(len(doc)):
    text.append(word_tokenize(doc[i]),is_approved[i])

freq = nltk.FreqDist(text[0])
freq.plot(cumulative=False)





#
