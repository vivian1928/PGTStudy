# import statements
import pandas as pd

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
	df_onehot = pd.DataFrame(index=range(48842), columns=df.columns[:-1])
	df_onehot['age'] = pd.get_dummies(df['age']).values.tolist()
	# for col in df.columns:
	# 	if col != 'class':
	# 		col_onehot = pd.get_dummies(df[col])
	# 		print(col_onehot)
	print(df_onehot['age'])

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	pass

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	pass

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	pass


df_adult = read_csv_1('data/adult.csv')
print(one_hot_encoding(df_adult))


