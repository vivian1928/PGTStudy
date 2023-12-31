# -*- coding: utf-8 -*-
"""adult.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_AdJLE9KNE_S9tF6qT5yM4ka9jaQLkm-
"""

# import statements
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.tree as tree
import sklearn.metrics as metrics

# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df.drop('fnlwgt', axis = 1, inplace=True)
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return [column for column in df]

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	num_vars = [column for column in df]
	return df[num_vars].isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	num_vars = [column for column in df]
	missing_list = df[num_vars].isnull().sum().tolist()
	column_missing = []
	for i in range(len(missing_list)):
		if missing_list[i] > 0:
			column_missing.append(num_vars[i])
	return column_missing

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
	return round((df['education'].value_counts()['Bachelors'] + \
			df['education'].value_counts()['Masters']) * 100 / df.shape[0],3)#.groupby('Bachelors').count()

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	return df.dropna().reset_index(drop=True)

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
	df_dropped = data_frame_without_missing_values(df)
	df_onehot = pd.DataFrame(index=range(num_rows(df_dropped)), columns=df_dropped.columns[:-1])
	for col in df.columns:
		if col != 'class':
			df_onehot[col] = pd.get_dummies(df_dropped[col]).values.tolist()
	return df_onehot

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	df_dropped = data_frame_without_missing_values(df)
	labelencoder = LabelEncoder()
	df_class = pd.DataFrame(index=range(num_rows(df_dropped)), columns=[df_dropped.columns[-1]])
	df_class['class'] = labelencoder.fit_transform(df_dropped['class'])
	return df_class

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	clf = tree.DecisionTreeClassifier(random_state = 0)
	clf.fit(X_train, y_train)
	y_hat = clf.predict(X_train)
	np_y_hat = np.array(y_hat)
	pd_y_hat = pd.Series(np_y_hat)
	return pd_y_hat
 
# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	return 1 - metrics.precision_score(y_true, y_pred, average=None)

import numpy as np
import sklearn.model_selection as model_select
df_adult = read_csv_1('data/adult.csv')
df_onehot = one_hot_encoding(df_adult)
df_class = label_encoding(df_adult)
x_pre = []
y_pre = []
for i in range(num_rows(df_onehot)):
  x_pre.append(np.hstack((df_onehot['age'][i], df_onehot['workclass'][i], df_onehot['education'][i], df_onehot['education-num'][i], df_onehot['marital-status'][i], df_onehot['occupation'][i]\
                        , df_onehot['relationship'][i], df_onehot['race'][i], df_onehot['sex'][i], df_onehot['capitalgain'][i], df_onehot['capitalloss'][i], df_onehot['hoursperweek'][i], df_onehot['native-country'][i])))
  y_pre.append(df_class['class'][i])
x_pre = np.array(x_pre)
y_pre = np.array(y_pre)

y_hat = dt_predict(x_pre, y_pre)
print(y_hat.shape)
print(dt_error_rate(y_hat, y_pre))
# for i in range(num_rows(df_onehot)):
#   print(df_onehot['age'][i])
#   print(df_onehot['workclass'][i])

X_train, X_test, y_train, y_test = model_select.train_test_split(x_pre, y_pre, random_state=0)
M_train = len( X_train )
M_test = len( X_test )
if ( DEBUGGING ):
    print('number of training instances = ' + str( M_train ))
    print('number of test instances = ' + str( M_test ))