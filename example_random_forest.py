from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
print(boston.head())

# Make arrays for doing Random Forest Regression--X,y.
proj = pd.read_csv('/Users/jif/Donors_choose/Projects.csv', encoding='utf-8', iterator=True, chunksize=100)
proj_chunk_one = proj.get_chunk(100)

school = pd.read_csv('/Users/jif/Donors_choose/Schools.csv', encoding='utf-8', iterator=True, chunksize=100)
school_chunk_one = school.get_chunk(100)
school_chunk_one = school_chunk_one.drop(['School Name','School Zip','School City','School County'], axis=1)

teacher = pd.read_csv('/Users/jif/Donors_choose/Teachers.csv', encoding='utf-8', iterator=True, chunksize=100)
teacher_chunk_one = teacher.get_chunk(100)

# Determine whether the project will get funded--
# change project will make(subject/object cost/use), how desperately class needs it(geographical location (school state, morphology)/school district (school district)/description of students (percentage of students free lunch)),
# gender, size of school (school district), number of projects submitted prior (transform of teacher first project posted date and count previous projects by teacher id), teacher prefix [Dr, Mr, Mrs, Ms, Teacher, N/A] (teacher prefix), teacher name (teacher id)

# Next, if it is useful, add more project columns.
# Proj Don't need columns:
# Proj Needed columns: 0, 1, 2

# Write SQL to transform Project, School, and Teacher tables to produce new report per project, teacher, and school.
# --Load report here--

# There are no scl_Y values
scl_X = school_chunk_one.values[:,:]

# There are no tchr_Y values
tchr_X = teacher_chunk_one.values[:,:]

X = chunk_one.values[:,:8]
y = chunk_one.values[:,16]
