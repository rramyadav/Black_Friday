#This file is used to preprocess data for machine learning
#dataset used is from analyticsvidhya.com hackdata practice problem
#data is known as "Black friday"
#https://datahack.analyticsvidhya.com/contest/black-friday/

#libary import

import pandas as pd
#libarary for missing data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#import numpy as np
from sklearn.model_selection import train_test_split
'''
Data prepossessing begin here
'''

#importing dataset
#dataset Train = file giveng for traing test Ml algo
#dataset_test = data for pridiction
dataset_train = pd.read_csv('train_AV_BF.csv')
#X = dataset_train.drop(dataset_train['Product_ID'],axis=1,inplace=True)
dataset_test = pd.read_csv('test_AV_BF.csv')


#X - is independent variable
#Y - is dependent variable

X = dataset_train.iloc[:,2:-1]
Y = dataset_train.iloc[:,-1]
Z = dataset_test.iloc[:,2:]
print(X.columns)
print (X.shape)
print (Z.shape)

# there are three startigies to remove Nan
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
#imputer_mean=Imputer(missing_values='NaN',strategy='mean',axis=0)
#but we will use fillna since nan = 0
#X = X.drop(X['Product_ID'],axis = 1, inplace = True)
X = X.fillna(0)
Z = Z.fillna(0)

#-------------------------------------------------------------------------------------
#Coding Categorial data
#-------------------------------------------------------------------------------------
labelencode_X = LabelEncoder()
#for i in X.columns:
# label encoding

#X = X.apply(LabelEncoder().fit)
#X[:,0:5] = labelencode_X.fit_transform(X[:,0:5])
#Z = Z.apply(LabelEncoder().transform)
for i in range(5):
    X.iloc[:, i] = labelencode_X.fit_transform(X.iloc[:, i])
    Z.iloc[:, i] = labelencode_X.fit_transform(Z.iloc[:, i])

#Using one hot encoder
#X = X.astype('int64')

one_HE = OneHotEncoder(categorical_features=[0,1,2,3,4,5])
X = one_HE.fit_transform(X).toarray()
Z = one_HE.fit_transform(Z).toarray()
# def labelencode(X):
#     X = X.apply(LabelEncoder().fit_transform)
#     one_HE = OneHotEncoder()
#     X = one_HE.fit_transform(X).toarray()
#     return X
#
# X = labelencode(X)
# Z = labelencode(Z)
#
print (X.shape)
print (Z.shape)


#-----------------------------------------------------------------------------
#          Spliting dataset
#--------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
