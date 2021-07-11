# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:39:14 2021

@author: Enias
"""


import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("C:\Program Files\Git\DataSets\insurance.csv")


X = dataset[["age","sex","bmi","children","smoker"]]
Y = dataset[["charges"]]

X = pd.get_dummies(data=X, drop_first=True)


model = LinearRegression()
model.fit(X,Y)



pickle.dump(model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


