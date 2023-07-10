# classify-iris-perceptron-error.py
# parsons/23-feb-2019
#
# Using a perceptron on the iris dataset, updating with the error
# correction rule.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

# Parameters
alpha     = 0.01

#
# Load and pre-process the data
#
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

# For the perceptron we rescale the data to be between 0 and 1.
max0 = 0.0
max1 = 0.0
for i in range(len(X)):
    if X[i][0] > max0:
        max0 = X[i][0]
    if X[i][1] > max1:
        max1 = X[i][1]

for i in range(len(X)):
    X[i][0] = float(X[i][0])/max0
    X[i][1] = float(X[i][1])/max1
    
#
# Here we build a perceptron.
#
# There are two inputs, so three weights (one per input and one for the bias).
# Pick weights as random numbers between 0 and 1
weights = [random.random(), random.random(), random.random()]

#
# Run error correction until the error rate is not dropping
#
errorRates = []
rateChanged = True
lastErrorRate = 1.0
while rateChanged:
    
    # Error correction for one epoch.
    #
    # For this one we are trying to distinguish one class from the other 2
    errors = 0
    for i in range(len(X)):
        sum = weights[0] + X[i][0]*weights[1] + X[i][1]*weights[2]
        
        # Real and desired outputs 
        if sum <= 0:
            output = 0
        else:
            output = 1

        # y = 0 is setosa, the easiest to classify
        if y[i] == 0:
            target = 0
        else:
            target = 1
                
        if target != output:
            errors = errors + 1
        
        # Update
        #
        # This is a bit of a nasty hack. A clean solution would treat
        # all the weights the same and have a single update. But that
        # means modifying X.
        weights[0] = weights[0] + alpha * (target - output)
        
        for j in [0, 1]:
            weights[j+1] = weights[j+1] + alpha * (target - output) * X[i][j]

    errorRate = float(errors)/len(X)
    errorRates.append(errorRate)

    if errorRate == lastErrorRate:
        rateChanged = False
    else:
        lastErrorRate = errorRate

print(errorRate)

#
# Now run the trained perceptron on the data.
#
# Of course we should really run this on data we did not use for
# training, but the point here is to see how well we have trained.
z = []
for i in range(len(X)):
    sum = weights[0] + X[i][0]*weights[1] + X[i][1]*weights[2]
    
    if sum <= 0:
        output = 0
    else:
        output = 1
        
    z.append(output)

#
# Now pllot the results, both the result of the perceptron and the
# original dataset for comparison.
#
# Result of perceptron
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30, c=z)
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Also plot the original data
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
plt.show()
