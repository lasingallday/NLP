# from sklearn.datasets import load_boston
# boston = load_boston()
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
df_empty['book_flag'] = np.where(df_empty['project_need_statement'].str.contains('book'),1,0)
df_empty['class_flag'] = np.where(df_empty['project_need_statement'].str.contains('class'),1,0)
df_empty['help_flag'] = np.where(df_empty['project_need_statement'].str.contains('help'),1,0)
df_empty['learning_flag'] = np.where(df_empty['project_need_statement'].str.contains('learning'),1,0)
df_empty['reading_flag'] = np.where(df_empty['project_need_statement'].str.contains('reading'),1,0)
df_empty['math_flag'] = np.where(df_empty['project_need_statement'].str.contains('math'),1,0)
df_empty['skills_flag'] = np.where(df_empty['project_need_statement'].str.contains('skills'),1,0)
df_empty['ipad_flag'] = np.where(df_empty['project_need_statement'].str.contains('ipad'),1,0)
df_empty['use_flag'] = np.where(df_empty['project_need_statement'].str.contains('use'),1,0)
df_empty['learn_flag'] = np.where(df_empty['project_need_statement'].str.contains('learn'),1,0)
df_empty['materials_flag'] = np.where(df_empty['project_need_statement'].str.contains('materials'),1,0)
df_empty['technology_flag'] = np.where(df_empty['project_need_statement'].str.contains('technology'),1,0)
df_empty['new_flag'] = np.where(df_empty['project_need_statement'].str.contains('new'),1,0)
df_empty['science_flag'] = np.where(df_empty['project_need_statement'].str.contains('science'),1,0)
df_empty['work_flag'] = np.where(df_empty['project_need_statement'].str.contains('work'),1,0)
df_empty['school_flag'] = np.where(df_empty['project_need_statement'].str.contains('school'),1,0)
df_empty['set_flag'] = np.where(df_empty['project_need_statement'].str.contains('set'),1,0)
df_empty['order_flag'] = np.where(df_empty['project_need_statement'].str.contains('order'),1,0)
df_empty['center_flag'] = np.where(df_empty['project_need_statement'].str.contains('center'),1,0)
df_empty['read_flag'] = np.where(df_empty['project_need_statement'].str.contains('read'),1,0)
df_empty['paper_flag'] = np.where(df_empty['project_need_statement'].str.contains('paper'),1,0)
df_empty['create_flag'] = np.where(df_empty['project_need_statement'].str.contains('create'),1,0)
df_empty['chromebooks_flag'] = np.where(df_empty['project_need_statement'].str.contains('chromebooks'),1,0)
df_empty['seating_flag'] = np.where(df_empty['project_need_statement'].str.contains('seating'),1,0)
df_empty['activities_flag'] = np.where(df_empty['project_need_statement'].str.contains('activities'),1,0)
df_empty['literacy_flag'] = np.where(df_empty['project_need_statement'].str.contains('literacy'),1,0)
df_empty['access_flag'] = np.where(df_empty['project_need_statement'].str.contains('access'),1,0)
df_empty['game_flag'] = np.where(df_empty['project_need_statement'].str.contains('game'),1,0)
df_empty['2_flag'] = np.where(df_empty['project_need_statement'].str.contains('2'),1,0)
df_empty['chair_flag'] = np.where(df_empty['project_need_statement'].str.contains('chair'),1,0)
df_empty['headphones_flag'] = np.where(df_empty['project_need_statement'].str.contains('headphones'),1,0)
df_empty['time_flag'] = np.where(df_empty['project_need_statement'].str.contains('time'),1,0)
df_empty['writing_flag'] = np.where(df_empty['project_need_statement'].str.contains('writing'),1,0)
df_empty['library_flag'] = np.where(df_empty['project_need_statement'].str.contains('library'),1,0)
df_empty['art_flag'] = np.where(df_empty['project_need_statement'].str.contains('art'),1,0)
df_empty['practice_flag'] = np.where(df_empty['project_need_statement'].str.contains('practice'),1,0)
df_empty['marker_flag'] = np.where(df_empty['project_need_statement'].str.contains('marker'),1,0)
df_empty['improve_flag'] = np.where(df_empty['project_need_statement'].str.contains('improve'),1,0)
df_empty['two_flag'] = np.where(df_empty['project_need_statement'].str.contains('two'),1,0)
df_empty['make_flag'] = np.where(df_empty['project_need_statement'].str.contains('make'),1,0)
df_empty['project_flag'] = np.where(df_empty['project_need_statement'].str.contains('project'),1,0)
df_empty['enhance_flag'] = np.where(df_empty['project_need_statement'].str.contains('enhance'),1,0)
df_empty['also_flag'] = np.where(df_empty['project_need_statement'].str.contains('also'),1,0)
df_empty['keep_flag'] = np.where(df_empty['project_need_statement'].str.contains('keep'),1,0)

# Hash function code snippet,
# myData=np.genfromtxt(filecsv, delimiter=",", dtype ="|a20" ,skip_header=1);
# le = preprocessing.LabelEncoder()
# for i in range(0,2):
#     myData[:,i] = le.fit_transform(myData[:,i])

# print(list(df_empty.columns.values))

# Make X and y like in load_boston (X has 47 columns). ADD columns 1, 2, 3 (with their hash functions)
X = df_empty.iloc[:,[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]
y = df_empty.iloc[:,10]
# print(X.head())


# Using the set of dictionary words and their likelihood of being funded, Predict whether passage will be funded.

# RUN LOGISTIC REGRESSION for whether the passage will be funded or not.
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)


# This is two for loops, with a dictionary created.
# It compares all words in train, with all words in dictionary. Then it adds the sentiment.
#t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
