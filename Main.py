# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:34:49 2018

Version 2.1.0, Last edited on Mon Dec  3 22:00:05 2018

CSE 447 Data Mining Final Project

@author: Keren Luo, Xinyu Gan
"""

import pandas as pd
import numpy as np
import re
import math
import datetime
from sklearn.compose import ColumnTransformer # NOTE: May Require scikit-learn Version 0.20.1.
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,FunctionTransformer,RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import cross_val_score

start = datetime.datetime.now()

#%% Read data
train = pd.read_csv('train_data.csv')
target = pd.read_csv('train_label.csv')
test = pd.read_csv('test_data.csv')
combine = [train, test]

#%% Text Cleaning
def clean(s):
    if type(s)==str:
        if s[0]=='<':
            match = re.findall(r'\>(.*?)\<',s)
        elif '<' in s:
            match = re.findall(r'\>(.*?)\<',s)
            string = re.search(r'(.*?)\<',s).group(1)
            match.insert(0,string)
        else:
            match = s.split('/')
    else:
        if math.isnan(s):
            match = 'null'
    for m in match:
        if m == ' ':
            match.remove(m)
    match = ' '.join(match)
    return match

#%% Text & Numerical Cleaning
for data in combine:
    new_desc = []
    for s in data['descrption']:
        new_desc.append(clean(s))
    data['descrption']=new_desc

    Percentile = np.percentile(data['price'],[0,25,50,75,100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3]
    DownLimit = Percentile[0]
    data.loc[data['price']<=DownLimit,'price'] = np.nan
    data.loc[data['price']>=UpLimit,'price'] = np.nan

#%% Building Sklearn Pipeline Classifier
numeric_features = ['price']
numeric_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                      ('scaler',RobustScaler()),
                                      ])

categorical_features = ['lvl1','lvl2','lvl3','type']
categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
                                          ('onehot',OneHotEncoder(handle_unknown='ignore')),
                                          ])

get_desc_data = FunctionTransformer(lambda x: x['descrption'], validate=False)
desc_features = ['descrption']
desc_transformer = Pipeline([
            ('selector', get_desc_data),
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
          ])

get_name_data = FunctionTransformer(lambda x: x['name'], validate=False)
name_features = ['name']
name_transformer = Pipeline([
            ('selector', get_name_data),
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
          ])

preprocessor = ColumnTransformer(transformers=[('num',numeric_transformer,numeric_features),
                                               ('cat',categorical_transformer,categorical_features),
                                               ('desc',desc_transformer,desc_features),
                                               ('name',name_transformer,name_features),
                                               ])

clf = Pipeline(steps=[('preprocessor',preprocessor),
                      ('classifier',LogisticRegression(solver='lbfgs')),
                      ])
#%% Fitting & Cross Validation
clf.fit(train,target['score'])

print(np.mean(cross_val_score(clf, train, target['score'], cv=5)))

#%% Results Saving
predict = clf.predict_proba(test)[:,1]

submission = pd.read_csv('submission.csv')
submission['score'] = predict
submission.to_csv('submission - Final.csv',index = False)

#%% Running Time
end = datetime.datetime.now()
print (end-start)
