from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, ngrams
from nltk.corpus import stopwords
from numpy import argmax, array
import csv


# Make arrays for doing Random Forest Regression--X,y.
# school = pd.read_csv('/Users/jif/Donors_choose/Schools.csv', encoding='utf-8', iterator=True, chunksize=100)
# school_chunk_1 = school.get_chunk(100)
# school_chunk_1 = school_chunk_1.drop(['School Name','School Zip','School City','School County'], axis=1)

raw = pd.read_csv('/Users/jif/Donors_choose/report_all_projects.csv', encoding='utf-8', sep=';', iterator=True, chunksize=10000)
raw_chunk_1 = raw.get_chunk(100)

df_empty = pd.DataFrame()
for chunk in raw:
    df_empty = pd.concat([df_empty,chunk])

print(list(df_empty.columns.values))

# Determine whether the project will get funded--
# change project will make(subject/object cost/use), how desperately class needs it(geographical location (school state, morphology)/school district (school district)/description of students (percentage of students free lunch)),
# gender, size of school (school district), number of projects submitted prior (transform of teacher first project posted date and count previous projects by teacher id), teacher prefix [Dr, Mr, Mrs, Ms, Teacher, N/A] (teacher prefix), teacher name (teacher id)

X = df_empty.values[:,3]
y = df_empty.values[:,9]


with open('/Users/jif/Donors_choose/fifty_most_common_words-proj_need_statement.csv','w') as f:
    commonwriter = csv.writer(f,delimiter=',',quotechar='"')
    for item in freq_n1.most_common(50):
        commonwriter.writerow([item[0], item[1]])


# Predict things

# Use only the set of dictionary words, with their likelihood of being funded, to predict whether passage will be funded.
# Make X and y like in load_boston

# Do a one-hot enocding for the top 50 common words. (Can also try binary in top 50/not in top 50)


# Run logistic regression for whether the passage will be funded or not.

# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
#t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
