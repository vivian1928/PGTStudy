#--
# bayes-hd.py
# parsons/25-feb-2017
# mumford/09-feb-2020
#
# The input data is the processed Cleveland data from the "Heart
# Diesease" dataset at the UCI Machine Learning repository:
#
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.
#--

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
# Make it possible to use the beta distribution
from scipy.stats import beta

#
# Define constants
#

datafilename = 'processed.cleveland.data' # input filename
age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file. 
num_features = 13

#
# Open and read data file in csv format
#
# After processing:
# 
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features ))
        target = np.empty(( num_samples,), dtype=np.int )
        i = 0
        for j, d in enumerate( indata ):
            ok = True
            for k in range(0,num_features): # If a feature has a missing value
                if ( d[k] == "?" ):         # we don't use that record.
                    ok = False
            if ( ok ):
                data[i] = np.asarray( d[:-1], dtype=np.float64 )
                target[i] = np.asarray( d[-1], dtype=np.int )
                i = i + 1
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

# How many records do we have?
num_samples = i
print("Number of samples:", num_samples)

# Adjust the size of data and target so that they only hold the values
# loaded from the CSV file
#
# This, elegant approach, is due to Adrian Salazar Gomez (2018/19):

data = data[:num_samples]
target = target[:num_samples]

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

##################################################################
#
# Question 3
#
# Classify with a decision tree
#

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print("Decision tree score:", clf.score(X_test, y_test))


#################################################################
#
# Question 4
#
# Build the Naive Bayes classifer
#

# Features to concentrate on are:
# sex
# cp
# fbs
# restecg
# exang
# slope
# ca

#
# Prior probability of each of the outcome values [0 to 4]
#

y_count = [0.0, 0.0, 0.0, 0.0, 0.0]
for i in range(len(y_train)):
    if y_train[i] == 0:
        y_count[0] += 1
    if y_train[i] == 1:
        y_count[1] += 1
    if y_train[i] == 2:
        y_count[2] += 1
    if y_train[i] == 3:
        y_count[3] += 1
    if y_train[i] == 4:
        y_count[4] += 1

# Count how many entries we have:
total = 0
for i in range(len(y_count)):
    total += y_count[i]

# Compute the prior value of each entry
priors = [0, 0, 0, 0, 0]
for i in range(len(y_count)):
    priors[i] = y_count[i]/total

#
# Conditional probabilities
#

# Compute the proabability of each value of cp given each value of num:
# cp has possible values 1, 2, 3, 4.
cp_num = [[0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0]]

# Count the number of entries with each value of cp, given a value of num
# cp is feature number 2 in the data
for i in range(len(X_train)):  
    if y_train[i] == 0:        # num = 0
        if X_train[i][2] == 1:
            cp_num[0][0] += 1
        if X_train[i][2] == 2:
            cp_num[0][1] += 1
        if X_train[i][2] == 3:
            cp_num[0][2] += 1
        if X_train[i][2] == 4:
            cp_num[0][3] += 1
 
    if y_train[i] == 1:        # num = 1
        if X_train[i][2] == 1:
            cp_num[1][0] += 1
        if X_train[i][2] == 2:
            cp_num[1][1] += 1
        if X_train[i][2] == 3:
            cp_num[1][2] += 1
        if X_train[i][2] == 4:
            cp_num[1][3] += 1
 
    if y_train[i] == 2:        # num = 2
        if X_train[i][2] == 1:
            cp_num[2][0] += 1
        if X_train[i][2] == 2:
            cp_num[2][1] += 1
        if X_train[i][2] == 3:
            cp_num[2][2] += 1
        if X_train[i][2] == 4:
            cp_num[2][3] += 1
 
    if y_train[i] == 3:        # num = 3
        if X_train[i][2] == 1:
            cp_num[3][0] += 1
        if X_train[i][2] == 2:
            cp_num[3][1] += 1
        if X_train[i][2] == 3:
            cp_num[3][2] += 1
        if X_train[i][2] == 4:
            cp_num[3][3] += 1
 
    if y_train[i] == 4:        # num = 4
        if X_train[i][2] == 1:
            cp_num[4][0] += 1
        if X_train[i][2] == 2:
            cp_num[4][1] += 1
        if X_train[i][2] == 3:
            cp_num[4][2] += 1
        if X_train[i][2] == 4:
            cp_num[4][3] += 1

# Now normalize to get probabilities:
for i in range(len(y_count)):
    for j in range(len(cp_num[1])):
        cp_num[i][j] = cp_num[i][j]/y_count[i]

#print cp_num

# Compute the proabability of each value of fbs given each value of num:
# fbs has possible values 1 and 0
fbs_num = [[0.0, 0.0],
           [0.0, 0.0],
           [0.0, 0.0],
           [0.0, 0.0],
           [0.0, 0.0]]
          
# Count the number of entries with each value of fbs, given a value of num
# fbs is feature number 5 in the data
for i in range(len(X_train)):  
    if y_train[i] == 0:        # num = 0
        if X_train[i][5] == 0:
            fbs_num[0][0] += 1
        if X_train[i][5] == 1:
            fbs_num[0][1] += 1

    if y_train[i] == 1:        # num = 1
        if X_train[i][5] == 0:
            fbs_num[1][0] += 1
        if X_train[i][5] == 1:
            fbs_num[1][1] += 1

    if y_train[i] == 2:        # num = 2
        if X_train[i][5] == 0:
            fbs_num[2][0] += 1
        if X_train[i][5] == 1:
            fbs_num[2][1] += 1
            
    if y_train[i] == 3:        # num = 3
        if X_train[i][5] == 0:
            fbs_num[3][0] += 1
        if X_train[i][5] == 1:
            fbs_num[3][1] += 1

    if y_train[i] == 4:        # num = 4
        if X_train[i][5] == 0:
            fbs_num[4][0] += 1
        if X_train[i][5] == 1:
            fbs_num[4][1] += 1

# Now normalize to get probabilities:
for i in range(len(y_count)):
    for j in range(len(fbs_num[0])):
        fbs_num[i][j] = fbs_num[i][j]/y_count[i]

#print fbs_num

#
# Now we can do some prediction
#

y_pred   = 0
score    = 0.0

for i in range(len(y_test)):
    # Moved prob_num from outside loop so that probabilities are refreshed for each test instance.
    prob_num = [1, 1, 1, 1, 1]
    # For each possible value of num, compute p(cp|num)p(fbs|num)p(num) 
    for j in range(len(y_count)): 
        prob_num[j] = prob_num[j] * cp_num[j][(int(X_test[i][2]) - 1)]
        prob_num[j] = prob_num[j] * fbs_num[j][int(X_test[i][5])]
        prob_num[j] = prob_num[j] * priors[j]
    # The element with the maximum value is the prediction
    y_pred = prob_num.index(max(prob_num))

    # Is this the right value?
    if y_pred == y_test[i]:
        score += 1
        
print("My Naive Bayes scores: ", score/len(y_test))

#################################################################
#
# Question 5
#
# Use the scikit-learn Naive Bayes classifier
#

nb = MultinomialNB()
nb = nb.fit(X_train, y_train)

print("Naive Bayes score:", nb.score(X_test, y_test))
