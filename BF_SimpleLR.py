# Preparing dataset
## program for simple linearregression
import pandas as pd
from DataPrePro import data_prepro as  dpp


dataset_train = pd.read_csv('train_AV_BF.csv')
dataset_train = dataset_train.fillna(0)
print(dataset_train.head())
dataset_train = dataset_train[dataset_train.Product_Category_1 < 19 ]

dataset_train_in = dataset_train.iloc[:,2:]

print(dataset_train.head())


dataset_test = pd.read_csv('test_AV_BF.csv')
dataset_test=dataset_test.fillna(0)
dataset_test_in=dataset_test.iloc[:,2:]
print(dataset_test_in.columns)
print(dataset_train_in.columns)

X_train, X_test, y_train, y_test=dpp(dataset_train_in)
X_1=dpp(dataset_test_in,option=False)#**NOTE Check option value correctly


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import explained_variance_score
#print(explained_variance_score(y_test, y_pred))
y2_ped = regressor.predict(X_1)
# print(len(dataset_test),'/n',len(dataset_train))
# print(y2_ped.shape)

solution = pd.read_csv('Sample_Submission.csv')
solution[['User_ID','Product_ID']]=dataset_test[['User_ID','Product_ID']]
solution['Purchase']= pd.DataFrame(y2_ped)
solution.to_csv('SLR-Solution2.csv',index=False)
