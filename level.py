# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:36:19 2018

@author: luoke
"""

import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.preprocessing import robust_scale
from sklearn.naive_bayes import MultinomialNB
#%%
data = pd.read_csv('train_data.csv')
target = pd.read_csv('train_label.csv')
test = pd.read_csv('test_data.csv')
combine = [data, test]
#%% Unique lvl1
lvl1 = list()
for l in data['lvl1']:
    if l not in lvl1:
        lvl1.append(l)


#%% Unique lvl2 & 3
index = dict()
ind = 0
for i in lvl1:
    lvl2 = list()
    for j in data[data['lvl1'] == i]['lvl2']:
        if j not in lvl2:
            lvl2.append(j)
    for k in lvl2:
        lvl3 = list()
        for m in data[data['lvl2'] == k]['lvl3']:
            if m not in lvl3:
                lvl3.append(m)
        for n in lvl3:
            index[ind] = n
            ind += 1
    
#%%
redup = dict()
for key, value in index.items():
    if value not in redup.values():
        redup[key] = value
diffKeys = set(index.keys()) - set(redup.keys())
duplicate = dict()
for key in diffKeys:
    duplicate[key] = index.get(key)
dup = list()
for d in duplicate.values():
    if d not in dup:
        dup.append(d)
        
#%%
pair = dict()
index = 0
for i in range(0, len(data)):
    lv3 = data.loc[i]['lvl3']
    if lv3 not in dup:
        if lv3 not in pair:
            pair[lv3] = index
            index += 1 
    elif type(lv3) == str:
        lv2 = data.loc[i]['lvl2']
        strcon = lv2 + '_' + lv3
        data['lvl3'][i] = strcon
        if strcon not in pair:
            pair[strcon] = index
            index += 1
    elif math.isnan(lv3):
        lv2 = data.loc[i]['lvl2']
        strcon = 'NaN_' + lv2
        data['lvl3'][i] = strcon
        if strcon not in pair:
            pair[strcon] = index
            index += 1
            
#%%
#data['lvl3'] = data['lvl3'].map(pair)

data['lvl3'] = data['lvl3'].map(pair)

#%% Type
list1 = list()
for i in range(0,len(data)):
    tt = data['type'][i]
    if type(tt) != str and math.isnan(data['type'][i]):
        list1.append(i)
for i in list1:
    data['type'][i] = 'n'
combine = [data, test]
data['type'] = data['type'].map({'international': 1, 'local': 0, 'n': np.random.randint(0,1)})

#%% price

data['price']=preprocessing.robust_scale(np.reshape(np.array(data['price']),(-1,1)))
test['price']=preprocessing.robust_scale(np.reshape(np.array(test['price']),(-1,1))) 


#%%Training

X_train = data[['lvl3','price','type']]

clf = MultinomialNB().fit(X_train,target['score'] )
predicted = clf.predict_proba(test[['lvl3','price','type']])[:,1]