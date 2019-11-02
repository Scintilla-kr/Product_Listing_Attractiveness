# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:24:18 2018

@author: luoke
"""
import pandas as pd
import re 
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


data = pd.read_csv('train_data.csv')
target = pd.read_csv('train_label.csv')
test = pd.read_csv('test_data.csv')
#%%
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

#%%
desc_train = []
desc_test = []
for s in data['descrption']:
    desc_train.append(clean(s))
    
for s in test['descrption']:
    desc_test.append(clean(s))   

data['new_descrption']=desc_train
test['new_descrption']=desc_test

 


#%% NB
#text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB()),])
#text_clf.fit(new_train,target['score'])
#
#predicted = text_clf.predict_proba(new_test)[:,1]

#%% SVM
text_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='modified_huber',penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None)),])
text_clf.fit(desc_train,target['score'])

predicted = text_clf.predict_proba(desc_test)[:,1]

#%% Count Vectorizer
#
#count_vect = CountVectorizer()
#
#X_train_counts = count_vect.fit_transform(desc_train)
#print(count_vect.get_feature_names())

#%% TFIDF
#
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#%% Merge features
#
#tf = X_train_tfidf.toarray()
#data['tfidf'] = tf
#np.concatenate((tf,pri_train))
#%%
#X_train = data[['tfidf','new_price']]
#clf = MultinomialNB().fit(X_train,target['score'] )
#clf = SGDClassifier(loss='modified_huber',penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None).fit(X_train,target['score'])

#%%
#X_new_counts = count_vect.transform(desc_test)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#test['tfidf']=list(X_new_tfidf)
##
#predicted = clf.predict_proba(X_new_tfidf)
#%%
#predicted_pri = clf.predict_proba(pri_test)[:,1]
