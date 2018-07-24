import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize, word_tokenize, ngrams

raw = pd.read_csv('/Users/jt/Documents/fun_projects/Projects.csv', encoding='utf-8', iterator=True, chunksize=100)
# 18 col--column header = project_id, school_id, teacher_id, teacher_project_posted_sequence, project_type, project_title, project_essay,
# project_short_description, project_need_statement, project_subject_category_tree, project_subject_subcategory_tree, project_grade_level_category,
# project_resource_category, project_cost, project_posted_date, project_expiration_date, project_current_status, project_fully_funded_date

# columns 6-8 are interesting texts.

chunk_one = raw.get_chunk(100)
# df_empty = pd.DataFrame()
# for chunk in raw:
#     # print(chunk)
#     df_empty = pd.concat([df_empty,chunk])
#
# df_empty.to_csv(r'/Users/jt/Desktop/small_file.csv', encoding='utf-8')

X = chunk_one.values[:,8]
y = chunk_one.values[:,16]
label_encoder = LabelEncoder()
integer_encoded_y = label_encoder.fit_transform(y)
doc_list = X.tolist()
response_list = integer_encoded_y.tolist()
train = zip(doc_list,response_list)

text = []
for i in range(len(doc_list)):
    text.append(word_tokenize(train[i][0]))

# This is two for loops.
# It splits up all words, in the process of creating a set of all words in the text.
# dictionary = set(word.lower().encode("utf-8") for passage in train for word in word_tokenize(passage[0]))
dictionary = list(word.lower().encode("utf-8") for passage in train for word in word_tokenize(passage[0]))

# t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]

freq_n1 = nltk.FreqDist(dictionary)
freq_n1.plot(cumulative=False)
# print(dictionary)
# d3 = ngramize(text,n=3)
# freq_n3 = nltk.FreqDist(d3)
# freq_n3.plot(cumulative=False)
