# from sklearn.datasets import load_boston
# boston = load_boston()
# from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def print_metrics(y_actual, y_predict):
    """Prints multiple metrics"""

    print("Accuracy:", (y_predict == y_actual).mean())
    print("Precision:", precision_score(y_actual, y_predict))
    print("Recall:", recall_score(y_actual, y_predict))
    print("F1-score:", f1_score(y_actual, y_predict))
    print("Matthews correlation coefficient:", matthews_corrcoef(y_actual, y_predict))
    print('\n')

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

# Encode hash functions (into integers).
le = preprocessing.LabelEncoder()
for i in range(2,4):
    df_empty.ix[:,i] = le.fit_transform(df_empty.ix[:,i])

# print(list(df_empty.columns.values))

# Load X and y like in load_boston (X has 49 columns).
X = df_empty.iloc[:,[2,3,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]
y = df_empty.iloc[:,10]
# print(X.head())

# Using the set of dictionary words and their likelihood of being funded, Predict whether passage will be funded.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# RUN LOGISTIC REGRESSION for whether the passage will be funded or not.
# To run this with a GridSearch for hyperparameters remove the fit function. Then use the LogisticRegression as
# a paremter in a GridSearchCV function.
# clf = LogisticRegression(random_state=0, solver='lbfgs',
#                          multi_class='multinomial').fit(X_train, y_train)
# GIVES ERROR "ValueError: Solver lbfgs supports only l2 penalties, got l1 penalty."
# logistic = LogisticRegression(random_state=0, solver='lbfgs',
#                          multi_class='multinomial')

# Create logistic regression model
# logistic = LogisticRegression()
# # Logistic Regression hyperparameters
# # Create regularization penalty space
# penalty = ['l1', 'l2']
# # Create regularization hyperparameter space
# C = np.logspace(0, 4, 10)
# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# Create SVC model. Support Vector Classification models take forever to train past 10,000 rows.
# svc = svm.SVC()
# # SVC hyperparameters
# parameters = {'C':[1, 10], 'kernel':('linear', 'rbf')}

# clf = GridSearchCV(logistic, param_grid=hyperparameters, cv=5, verbose=0)

# Fit grid search
# best_model = clf.fit(X_train,y_train)
# # View best SVC hyperparameters
# # print('Best SVC Regression Paramters:', best_model.best_params_)
# # View best Logistic Regression hyperparameters
# # After that, use decision trees to predict the weak classifiers and aggregate the DT's to make a
# # prediction (i.e. perform an ensemble method prediction)
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:', best_model.best_estimator_.get_params()['C'])

# print_metrics(y_test, clf.predict(X_test))
# ypred = clf.predict(X_test)
# conf_mat = confusion_matrix(y_test,ypred)
# print(conf_mat)

# RUN RANDOM FOREST CLASSIFICATION for whether the passage will be funded or not.
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                            random_state=0)
clf.fit(X_train, y_train)

print_metrics(y_test, clf.predict(X_test))

# The 14th element (column 22) is most important. It incidates whether or not "technology" is in the project need statement.
# The 2nd element is second most important. It is either teacher_id or school_id.
# print(clf.feature_importances_)
