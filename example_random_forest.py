from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Determine whether the project will get funded--
# change project will make(subject/object cost/use), how desperately class needs it(geographical location (school state, morphology)/school district (school district)/description of students (percentage of students free lunch)),
# gender, size of school (school district), number of projects submitted prior (transform of teacher first project posted date and count previous projects by teacher id), teacher prefix [Dr, Mr, Mrs, Ms, Teacher, N/A] (teacher prefix), teacher name (teacher id)


# There are no scl_Y values
# scl_X = school_chunk_1.values[:,:]

# # There are no tchr_Y values
# tchr_X = teacher_chunk_1.values[:,:]

X = raw_chunk_1.values[:,3]
y = raw_chunk_1.values[:,9]
