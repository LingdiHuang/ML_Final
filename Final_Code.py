import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import time
from sklearn import preprocessing
import re
import random
from sklearn import svm
import seaborn as sns

##Import Data
url = 'https://raw.githubusercontent.com/LingdiHuang/ML_Final/master/df_sample.csv'
df = pd.read_csv(url,index_col=0,parse_dates=[0])

x = df.drop('V120',axis = 1)
y = df[['V120']]

x = x.drop('Unnamed: 0',axis = 1)


## Model
train_error_lasso1 = []
test_error_lasso1 = []
cross_lasso1_scores1 = []

train_error_ridge1 = []
test_error_ridge1 = []
cross_ridge_scores1 = []

train_error_tree1 = []
test_error_tree1 = []

train_error_log1 = []
test_error_log1 = []

train_error_svm1 = []
test_error_svm1 = []
cross_svm_scores1 = []

for i in range(50):
    print(i,end = ' ')
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5, random_state=i)
    ##Lasso
    log = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000)
    grid_lasso={"penalty":["l1"],"C":np.logspace(-2,2,5)}
    log_cv = GridSearchCV(log,grid_lasso,cv=10,n_jobs=-1,return_train_score=True)
    log_fit = log_cv.fit(x_train,y_train.values.ravel())
    pred_test_l = log_fit.predict(x_test)
    pred_train_l = log_fit.predict(x_train)
    test_error_lasso1.append(1-accuracy_score(y_test,pred_test_l))
    train_error_lasso1.append(1-accuracy_score(y_train,pred_train_l))
    cross_lasso1_scores1.append(1-log_cv.cv_results_['mean_test_score'].max())
    
    ##Ridge
    log = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000)
    grid_ridge={"penalty":["l2"],"C":np.logspace(-2,2,5)}
    log_cv = GridSearchCV(log,grid_ridge,cv=10,n_jobs=-1,return_train_score=True)
    log_fit = log_cv.fit(x_train,y_train.values.ravel())
    pred_test_r = log_fit.predict(x_test)
    pred_train_r = log_fit.predict(x_train)
    test_error_ridge1.append(1-accuracy_score(y_test,pred_test_r))
    train_error_ridge1.append(1-accuracy_score(y_train,pred_train_r))
    cross_ridge_scores1.append(1-log_cv.cv_results_['mean_test_score'].max())
    
    ##Random Forrest
    model = RandomForestClassifier(n_estimators=300,bootstrap = True,max_features = 'sqrt')
    tree_fit = model.fit(x_train,y_train.values.ravel())
    pred_test_t = tree_fit.predict(x_test)
    pred_train_t = tree_fit.predict(x_train)
    test_error_tree1.append(1-accuracy_score(y_test,pred_test_t))
    train_error_tree1.append(1-accuracy_score(y_train,pred_train_t))   
    
    ##Logistics
    logistic = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000,C = 1e4)
    log_fit_adj = logistic.fit(x_train,y_train.values.ravel())
    pred_test_lo = log_fit.predict(x_test)
    pred_train_lo = log_fit.predict(x_train)
    test_error_log1_adj.append(1-accuracy_score(y_test,pred_test_lo))
    train_error_log1_adj.append(1-accuracy_score(y_train,pred_train_lo))
    
    ##svm
    grid_para_svm = [{"C":np.logspace(-2,2,5),'gamma': np.logspace(-2,2,5)}]
    svm_cv = GridSearchCV(svm.SVC(kernel='rbf'), grid_para_svm, cv = 10, return_train_score=True,  n_jobs=-1)
    svm_fit = svm_cv.fit(x_train,y_train.values.ravel())
    pred_test_sv = svm_fit.predict(x_test)
    pred_train_sv = svm_fit.predict(x_train)
    test_error_svm1.append(1-accuracy_score(y_test,pred_test_sv))
    train_error_svm1.append(1-accuracy_score(y_train,pred_train_sv))
    cross_svm_scores1.append(1-svm_cv.cv_results_['mean_test_score'].max())


##Plot Error Rate

df_err_sample1 = pd.DataFrame({'train_error_Lasso':train_error_lasso1,'train_error_Ridge':train_error_ridge1,
                       'train_error_Logistic':train_error_log1_adj,'train_error_svm':train_error_svm1,
                       'train_error_Tree':train_error_tree1,'test_error_Logistic':test_error_log1_adj,
                       'test_error_svm':test_error_svm1,'test_error_Lasso':test_error_lasso1,
                       'test_error_Ridge':test_error_ridge1,'test_error_Tree':test_error_tree1,})
df_err_cor_sample1 = pd.DataFrame({'cor_err_svm':cross_svm_scores1,'cor_err_Lasso':cross_lasso1_scores1,
                           'cor_err_Ridge':cross_ridge_scores1})

df_new = df_err_sample1.stack().reset_index()
df_new['Model'] = df_new['level_1'].apply(lambda x : x.split('_')[-1])
df_new['set'] = df_new['level_1'].apply(lambda x : ' '.join(x.split('_')[:-1]))
df_new.rename(columns={0:'Error_rate'}, inplace=True)

df_train = df_new[(df_new['set']=="train error")]
df_test = df_new[(df_new['set']=="test error")]



ax = sns.boxplot(x=df_new['Model'], y = df_new['Error_rate'], data=df_new, hue=df_new['set'],showmeans=True, meanline=True, palette = "Set3")
labels = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
titles = ax.set_title('Error Rates Split by 0.5n')

##Plot CV Error Rate
df_new_cor = df_err_cor_sample1.stack().reset_index()
df_new_cor['Model'] = df_new_cor['level_1'].apply(lambda x : x.split('_')[-1])
df_new_cor.rename(columns={0:'Error_rate'}, inplace=True)

ax = sns.boxplot(x=df_new_cor['Model'], y = df_new_cor['Error_rate'], data=df_new_cor, showmeans=True, meanline=True, palette = "Set3")
labels = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
titles = ax.set_title('Minimum Cross-Validation Error Rate (Split by 0.5n)')


##########################0.9n########################################################################

train_error_lasso2 = []
test_error_lasso2 = []
cross_lasso1_scores2 = []

train_error_ridge2 = []
test_error_ridge2 = []
cross_ridge_scores2 = []

train_error_tree2 = []
test_error_tree2 = []

train_error_log2 = []
test_error_log2 = []

train_error_svm2 = []
test_error_svm2 = []
cross_svm_scores2 = []

for i in range(50):
    print(i,end = ' ')
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state=i)
    ##Lasso
    log = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000)
    grid_lasso={"penalty":["l1"],"C":np.logspace(-2,2,5)}
    log_cv = GridSearchCV(log,grid_lasso,cv=10,n_jobs=-1,return_train_score=True)
    log_fit = log_cv.fit(x_train,y_train.values.ravel())
    pred_test_l = log_fit.predict(x_test)
    pred_train_l = log_fit.predict(x_train)
    test_error_lasso2.append(1-accuracy_score(y_test,pred_test_l))
    train_error_lasso2.append(1-accuracy_score(y_train,pred_train_l))
    cross_lasso1_scores2.append(1-log_cv.cv_results_['mean_test_score'].max())
    
    ##Ridge
    log = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000)
    grid_ridge={"penalty":["l2"],"C":np.logspace(-2,2,5)}
    log_cv = GridSearchCV(log,grid_ridge,cv=10,n_jobs=-1,return_train_score=True)
    log_fit = log_cv.fit(x_train,y_train.values.ravel())
    pred_test_r = log_fit.predict(x_test)
    pred_train_r = log_fit.predict(x_train)
    test_error_ridge2.append(1-accuracy_score(y_test,pred_test_r))
    train_error_ridge2.append(1-accuracy_score(y_train,pred_train_r))
    cross_ridge_scores2.append(1-log_cv.cv_results_['mean_test_score'].max())
    
    ##Random Forrest
    model = RandomForestClassifier(n_estimators=300,bootstrap = True,max_features = 'sqrt')
    tree_fit = model.fit(x_train,y_train.values.ravel())
    pred_test_t = tree_fit.predict(x_test)
    pred_train_t = tree_fit.predict(x_train)
    test_error_tree2.append(1-accuracy_score(y_test,pred_test_t))
    train_error_tree2.append(1-accuracy_score(y_train,pred_train_t))   
    
    ##Logistics
    logistic = LogisticRegression(solver='liblinear', multi_class='auto',max_iter = 2000, C=1e4)
    log_fit_adj = logistic.fit(x_train,y_train.values.ravel())
    pred_test_lo = log_fit_adj.predict(x_test)
    pred_train_lo = log_fit_adj.predict(x_train)
    test_error_log2_adj.append(1-accuracy_score(y_test,pred_test_lo))
    train_error_log2_adj.append(1-accuracy_score(y_train,pred_train_lo))
    
    ##svm
    grid_para_svm = [{"C":np.logspace(-2,2,5),'gamma': np.logspace(-2,2,5)}]
    svm_cv = GridSearchCV(svm.SVC(kernel='rbf'), grid_para_svm, cv = 10, return_train_score=True,  n_jobs=-1)
    svm_fit = svm_cv.fit(x_train,y_train.values.ravel())
    pred_test_sv = svm_fit.predict(x_test)
    pred_train_sv = svm_fit.predict(x_train)
    test_error_svm2.append(1-accuracy_score(y_test,pred_test_sv))
    train_error_svm2.append(1-accuracy_score(y_train,pred_train_sv))
    cross_svm_scores2.append(1-svm_cv.cv_results_['mean_test_score'].max())


#####Plot For Error Rate
df_err_sample2 = pd.DataFrame({'train_error_Lasso':train_error_lasso2,'train_error_Ridge':train_error_ridge2,
                       'train_error_Logistic':train_error_log2_adj,'train_error_svm':train_error_svm2,
                       'train_error_Tree':train_error_tree2,'test_error_Logistic':test_error_log2_adj,
                       'test_error_svm':test_error_svm2,'test_error_Lasso':test_error_lasso2,
                       'test_error_Ridge':test_error_ridge2,'test_error_Tree':test_error_tree2,})
df_err_cor_sample2 = pd.DataFrame({'cor_err_svm':cross_svm_scores2,'cor_err_Lasso':cross_lasso1_scores2,
                           'cor_err_Ridge':cross_ridge_scores2})


df_new2 = df_err_sample2.stack().reset_index()
df_new2['Model'] = df_new2['level_1'].apply(lambda x : x.split('_')[-1])
df_new2['set'] = df_new2['level_1'].apply(lambda x : ' '.join(x.split('_')[:-1]))
df_new2.rename(columns={0:'Error_rate'}, inplace=True)

df_train = df_new2[(df_new2['set']=="train error")]
df_test = df_new2[(df_new2['set']=="test error")]

ax = sns.boxplot(x=df_new2['Model'], y = df_new2['Error_rate'], data=df_new2, hue=df_new2['set'],showmeans=True, meanline=True, palette = "Set3")
labels = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
titles = ax.set_title('Error Rates Split by 0.9n')


##Plot for CV Erorr Rate

df_new_cor2 = df_err_cor_sample2.stack().reset_index()
df_new_cor2['Model'] = df_new_cor2['level_1'].apply(lambda x : x.split('_')[-1])
df_new_cor2.rename(columns={0:'Error_rate'}, inplace=True)


ax = sns.boxplot(x=df_new_cor2['Model'], y = df_new_cor2['Error_rate'], data=df_new_cor2, showmeans=True, meanline=True, palette = "Set3")
labels = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
titles = ax.set_title('Minimum Cross-Validation Error Rate (Split by 0.9n)')










































