# from sklearn.datasets import load_boston
# boston = load_boston()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, ngrams
from nltk.corpus import stopwords
from numpy import argmax, array
import csv

# Determine whether the project will get funded--
# change project will make(subject/object cost/use), how desperately class needs it(geographical location (school state, morphology)/school district (school district)/description of students (percentage of students free lunch)),
# gender, size of school (school district), number of projects submitted prior (transform of teacher first project posted date and count previous projects by teacher id), teacher prefix [Dr, Mr, Mrs, Ms, Teacher, N/A] (teacher prefix), teacher name (teacher id)


# Make arrays for doing Random Forest Regression--X,y.
raw = pd.read_csv('/Users/jif/Donors_choose/fifty_most_common_words-proj_need_statement.csv', encoding='utf-8', sep=',')

# print(list(raw.columns.values))

# Make X and y like in load_boston
X = raw.values[:,0]
# y = raw.values[:,2]


# Using the set of dictionary words and their likelihood of being funded, Predict whether passage will be funded.

# Do a one-hot encoding for the top 50 common words.
# MANUAL ONE-HOT ENCODING
# define a mapping of chars to integers
# char_to_int = dict((row[0],index) for index,row in raw.iterrows())
# # integer encode input data
# integer_encoded = [char_to_int[char] for char in X]
# # one hot encode
# onehot_encoded = list()
# for value in integer_encoded:
#     word = [0 for _ in range(len(X))]
#     word[value] = 1
#     onehot_encoded.append(word)

# SKLEARN ONE-HOT ENCODING
values = array(X)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode (integer_encoded is the mapped numbers for string values,
# onehot_encoded is the resultant is the bit encodings for the integer_encoded numbers).
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# Set classes, in-top-50 and not-in-top-50. (Optional)

# RUN LOGISTIC REGRESSION for whether the passage will be funded or not. (Add one-hot encoded values?)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)


# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
#t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
