# from sklearn.datasets import load_boston
# boston = load_boston()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, ngrams
from nltk.corpus import stopwords
import numpy as np
import csv

def ngramize(texts, n):
    output=[]
    for text in texts:
        output += ngrams(text,n)
    return output

# Determine whether the project will get funded--
# change project will make(subject/object cost/use), how desperately class needs it(geographical location (school state, morphology)/school district (school district)/description of students (percentage of students free lunch)),
# gender, size of school (school district), number of projects submitted prior (transform of teacher first project posted date and count previous projects by teacher id), teacher prefix [Dr, Mr, Mrs, Ms, Teacher, N/A] (teacher prefix), teacher name (teacher id)


# Make arrays for doing Random Forest Regression--X,y.
raw = pd.read_csv('/Users/jif/Donors_choose/report_all_projects_12302018.csv', encoding='utf-8', sep=',', iterator=True, chunksize=10000)
raw_chunk_1 = raw.get_chunk(100)

df_empty = pd.DataFrame()
for chunk in raw:
    df_empty = pd.concat([df_empty,chunk])

# Add flags for 50 most common words (actually 44, since one "word" is a null value and five words are duplicates)
df_empty['book_flag'] = np.where(df_empty['project_need_statement'].str.contains('book'),'True','False')
df_empty['class_flag'] = np.where(df_empty['project_need_statement'].str.contains('class'),'True','False')
df_empty['help_flag'] = np.where(df_empty['project_need_statement'].str.contains('help'),'True','False')
df_empty['learning_flag'] = np.where(df_empty['project_need_statement'].str.contains('learning'),'True','False')
df_empty['reading_flag'] = np.where(df_empty['project_need_statement'].str.contains('reading'),'True','False')
df_empty['math_flag'] = np.where(df_empty['project_need_statement'].str.contains('math'),'True','False')
df_empty['skills_flag'] = np.where(df_empty['project_need_statement'].str.contains('skills'),'True','False')
df_empty['ipad_flag'] = np.where(df_empty['project_need_statement'].str.contains('ipad'),'True','False')
df_empty['use_flag'] = np.where(df_empty['project_need_statement'].str.contains('use'),'True','False')
df_empty['learn_flag'] = np.where(df_empty['project_need_statement'].str.contains('learn'),'True','False')
df_empty['materials_flag'] = np.where(df_empty['project_need_statement'].str.contains('materials'),'True','False')
df_empty['technology_flag'] = np.where(df_empty['project_need_statement'].str.contains('technology'),'True','False')
df_empty['new_flag'] = np.where(df_empty['project_need_statement'].str.contains('new'),'True','False')
df_empty['science_flag'] = np.where(df_empty['project_need_statement'].str.contains('science'),'True','False')
df_empty['work_flag'] = np.where(df_empty['project_need_statement'].str.contains('work'),'True','False')
df_empty['school_flag'] = np.where(df_empty['project_need_statement'].str.contains('school'),'True','False')
df_empty['set_flag'] = np.where(df_empty['project_need_statement'].str.contains('set'),'True','False')
df_empty['order_flag'] = np.where(df_empty['project_need_statement'].str.contains('order'),'True','False')
df_empty['center_flag'] = np.where(df_empty['project_need_statement'].str.contains('center'),'True','False')
df_empty['read_flag'] = np.where(df_empty['project_need_statement'].str.contains('read'),'True','False')
df_empty['paper_flag'] = np.where(df_empty['project_need_statement'].str.contains('paper'),'True','False')
df_empty['create_flag'] = np.where(df_empty['project_need_statement'].str.contains('create'),'True','False')
df_empty['chromebooks_flag'] = np.where(df_empty['project_need_statement'].str.contains('chromebooks'),'True','False')
df_empty['seating_flag'] = np.where(df_empty['project_need_statement'].str.contains('seating'),'True','False')
df_empty['activities_flag'] = np.where(df_empty['project_need_statement'].str.contains('activities'),'True','False')
df_empty['literacy_flag'] = np.where(df_empty['project_need_statement'].str.contains('literacy'),'True','False')
df_empty['access_flag'] = np.where(df_empty['project_need_statement'].str.contains('access'),'True','False')
df_empty['game_flag'] = np.where(df_empty['project_need_statement'].str.contains('game'),'True','False')
df_empty['2_flag'] = np.where(df_empty['project_need_statement'].str.contains('2'),'True','False')
df_empty['chair_flag'] = np.where(df_empty['project_need_statement'].str.contains('chair'),'True','False')
df_empty['headphones_flag'] = np.where(df_empty['project_need_statement'].str.contains('headphones'),'True','False')
df_empty['time_flag'] = np.where(df_empty['project_need_statement'].str.contains('time'),'True','False')
df_empty['writing_flag'] = np.where(df_empty['project_need_statement'].str.contains('writing'),'True','False')
df_empty['library_flag'] = np.where(df_empty['project_need_statement'].str.contains('library'),'True','False')
df_empty['art_flag'] = np.where(df_empty['project_need_statement'].str.contains('art'),'True','False')
df_empty['practice_flag'] = np.where(df_empty['project_need_statement'].str.contains('practice'),'True','False')
df_empty['marker_flag'] = np.where(df_empty['project_need_statement'].str.contains('marker'),'True','False')
df_empty['improve_flag'] = np.where(df_empty['project_need_statement'].str.contains('improve'),'True','False')
df_empty['two_flag'] = np.where(df_empty['project_need_statement'].str.contains('two'),'True','False')
df_empty['make_flag'] = np.where(df_empty['project_need_statement'].str.contains('make'),'True','False')
df_empty['project_flag'] = np.where(df_empty['project_need_statement'].str.contains('project'),'True','False')
df_empty['enhance_flag'] = np.where(df_empty['project_need_statement'].str.contains('enhance'),'True','False')
df_empty['also_flag'] = np.where(df_empty['project_need_statement'].str.contains('also'),'True','False')
df_empty['keep_flag'] = np.where(df_empty['project_need_statement'].str.contains('keep'),'True','False')

# print(list(df_empty.columns.values))

# Make X and y like in load_boston (48 columns)
X = df_empty.iloc[:,[1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]
y = df_empty.values[:,10]



print(X.head())

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
