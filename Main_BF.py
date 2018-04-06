# Preparing dataset
import pandas as pd
from DataPrePro import data_prepro as  dpp


dataset_train = pd.read_csv('train_AV_BF.csv')
dataset_test = pd.read_csv('test_AV_BF.csv')

X_train, X_test, y_train, y_test=dpp(dataset_train)

print(X_train.shape)