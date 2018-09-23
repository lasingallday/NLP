# from sklearn.ensemble import RandomForestRegressor
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, ngrams
from nltk.corpus import stopwords

# boston = load_boston()
def ngramize(texts, n):
    output=[]
    for text in texts:
        output += ngrams(text,n)
    return output

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

# Set up X like train (with y).
doc_list = X.tolist()
response_list = y.tolist()
train = zip(doc_list,response_list)
text = []

for i in range(len(train)):
    # print(i," ",train[i][0])
    if train[i][0] == train[i][0]:
        # print("list is empty")
        text.append(word_tokenize(train[i][0]))

dictionary = []
for passage in train:
    if passage[0] == passage[0]:
        for word in word_tokenize(passage[0]):
            # print(word.lower())
            dictionary.append(word.lower().encode("utf-8"))

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
# dictionary = list(word.lower().encode("utf-8") for passage in train for word in word_tokenize(passage[0]))
# dictionary = dictionary[:100]

# Eliminate stop words
stop_words = set(stopwords.words('english'))
filtered_sentence = [w for w in dictionary if not w in stop_words]

# Eliminate punctuation
punct_words = set(['.',',','?','!','\"','\'',':',';','\\','&'])
proj_specific_stop_words = set(['need','students'])
final_sentence = [w for w in filtered_sentence if not w in punct_words and w not in proj_specific_stop_words]

# Can use dictionary or filtered_sentence as a parameter here.
freq_n1 = nltk.FreqDist(final_sentence)
# print(freq_n1)
# print(text)

######################################################
# Frequency distribution for bigrams, n-grams.
######################################################
# This creates n-grams (n=3) for sentences that make up our textself.
# Each sentence is a description of why a donation is needed for a particular project.
#d3 = ngramize(text,n=3)
#freq_n3 = nltk.FreqDist(d3)
freq_n1.plot(50, cumulative=False, title='Most Common Words')

# Predict things

dictionary = []
for passage in train:
    if passage[0] == passage[0]:
        for word in word_tokenize(passage[0]):
            # print(word.lower())
            dictionary.append(word.lower().encode("utf-8"))

# Use only the set of dictionary words, with their likelihood of being funded, to predict whether passage will be funded.


# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
