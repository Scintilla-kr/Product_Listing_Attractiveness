# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:51:05 2018

@author: Keren Luo
"""

#%%
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing

#%%
train = pd.read_csv('train_data.csv')
target = pd.read_csv('train_label.csv')
test = pd.read_csv('test_data.csv')
combine = [train, test]
#%%
for data in combine:
    Percentile = np.percentile(data['price'],[0,25,50,75,100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3]+IQR*1.5
    DownLimit = Percentile[0]
    data.loc[data['price']<=DownLimit,'price'] = np.nan
    data.loc[data['price']>=UpLimit,'price'] = np.nan
    mean_price = np.mean(data['price'])
    data.loc[data['price'].isnull(),'price'] = mean_price
    data['price'] = preprocessing.scale(np.reshape(np.array(data['price']),(-1,1)))
#%%



