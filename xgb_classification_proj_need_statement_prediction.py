# from sklearn.datasets import load_boston
# boston = load_boston()
# from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, cross_validation
# from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV # Use with Logistic Regression and SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef, mean_absolute_error, accuracy_score, roc_auc_score
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

def modelfit(alg, dtrain, dtest, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values, label=dtest[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtest_predictions = alg.predict(dtest[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]

    #Print model report:
    print "\nModel Report"
    print "Accuracy (Train): %.4g" % accuracy_score(dtrain[target].values, dtrain_predictions)
    # print "Accuracy (Test): %.4g" % accuracy_score(dtest[target].values, dtest_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(dtrain[target], dtrain_predprob)
    print "AUC Score (Test): %f" % roc_auc_score(dtest[target], dtest_predprob)

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


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
X = df_empty.iloc[:,[2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]
y = df_empty.iloc[:,10]
# print(X.head())

# Using the set of dictionary words and their likelihood of being funded, Predict whether passage will be funded.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size=0.1)

# This dataframe, df, is used for XGBClassifier.
# xgb_train = df_empty[200000:300000].iloc[:,[2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]
# xgb_test = df_empty.loc[200000:210000].iloc[:,[2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]]

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV

# imp = preprocessing.Imputer()
#
# X_train = imp.fit_transform(X_train)
# X_test = imp.transform(X_test)
#
# # RUN GRADIENT BOOSTED DECISION TREE for improving error.
# xgb = XGBClassifier(max_depth=10, n_estimators=1000)
# # Add silent=True to avoid printing out updates with each cycle
# xgb.fit(X_train, y_train, early_stopping_rounds=5,
#         eval_set=[(X_test, y_test)], verbose=False)
#
# # make predictions
# preds = xgb.predict(X_test)
# # print("Mean Absolute Error : " + str(mean_absolute_error(preds, y_test)))
# print_metrics(y_test, preds)

#Choose all predictors except target & IDcols
predictors = [x for x in X_train.columns if x not in ['flag_project_funded']]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.65,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

# Fit XGBClassifier and get accuracy and ROC scores
# modelfit(xgb1, X_train, X_test, predictors, 'flag_project_funded')

# Plot XGBClassifier feature importances
target = 'flag_project_funded'
xgb_param = xgb1.get_xgb_params()
xgtrain = xgb.DMatrix(X_train[predictors].values, label=X_train[target].values)
xgtest = xgb.DMatrix(X_test[predictors].values, label=X_test[target].values)
params = {"objective":"binary:logistic",'colsample_bytree': 0.8,'learning_rate': 0.1,
                'max_depth': 5, 'gamma': 0}

xg_reg = xgb.train(params=params, dtrain=xgtrain, num_boost_round=10)
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
