# -*- coding: utf-8 -*-
"""wholesale_customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14-SkaFCLwAb1-d76scan1dbqWsTSYG7k
"""

# Part 2: Cluster Analysis
# import statements
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df.drop(labels=['Channel', 'Region'], axis = 1, inplace=True)
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	sr_mean = df.mean().round().astype(int)
	sr_mean.name = 'mean'
	sr_std = df.std().round().astype(int)
	sr_std.name = 'std'
	sr_min = df.min()
	sr_min.name = 'min'
	sr_max = df.max()
	sr_max.name = 'max'
	return pd.concat([sr_mean, sr_std, sr_min, sr_max], axis=1)

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	return (df - df.mean()) / df.std()

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
	km=cluster.KMeans(n_clusters=k)
	km.fit(df)
	return pd.Series(km.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	kmpp=cluster.KMeans(n_clusters=k,init='k-means++')
	kmpp.fit(df)
	return pd.Series(kmpp.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	ac=cluster.AgglomerativeClustering(n_clusters=k,linkage='average',affinity='euclidean')
	ac.fit(df)
	return pd.Series(ac.labels_)

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X,y):
	return metrics.silhouette_score(X,y,metric='euclidean')

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	rdf=pd.DataFrame(columns=['Algorithm','data','k','Silhouette Score'])
	df_standard=standardize(df)
	for k in [3, 5, 10]:
		for j in range(10):
			km_orig_y=kmeans(df,k)
			km_stand_y=kmeans(df_standard,k)
		ag_orig_y=agglomerative(df,k)
		ag_stand_y=agglomerative(df_standard,k)
		km_orig_y_cs=clustering_score(df, km_orig_y)
		ag_orig_y_cs=clustering_score(df, ag_orig_y)
		km_stand_y_cs=clustering_score(df_standard, km_stand_y)
		ag_stand_y_cs=clustering_score(df_standard, ag_stand_y)
		rdf=rdf.append({'Algorithm':'Kmeans','data':'Original','k':k,'Silhouette Score':km_orig_y_cs},ignore_index=True)
		rdf=rdf.append({'Algorithm':'Agglomerative','data':'Original','k':k,'Silhouette Score':ag_orig_y_cs},ignore_index=True)
		rdf=rdf.append({'Algorithm':'Kmeans','data':'Standardized','k':k,'Silhouette Score':km_stand_y_cs},ignore_index=True)
		rdf=rdf.append({'Algorithm':'Agglomerative','data':'Standardized','k':k,'Silhouette Score':ag_stand_y_cs},ignore_index=True)
	return rdf

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	pass



#######################-----------main-----------#########################
def num_rows(df):
	return df.shape[0]

if __name__ == '__main__':
	df_customer = read_csv_2('wholesale_customers.csv')
	# df_standard = standardize(df_customer)
	# y=agglomerative(df_standard, 3)
	# print(clustering_score(df_standard, y))
	rdf=cluster_evaluation(df_customer)
	print(best_clustering_score(rdf))