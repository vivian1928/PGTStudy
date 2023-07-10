# classify-iris-perceptron.py
# parsons/23-feb-2019
#
# Using a perceptron on the iris dataset. The perceptron uses the
# generalised delta rule and a sigmoid transfer function.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

# Parameters
alpha = 0.1

# Function to implement the weighted sum for two inputs and a list of
# three weights.
def weightedSum(weights, inputs):
    sum = weights[0] + inputs[0]*weights[1] + inputs[1]*weights[2]

    return sum

# Function to implement a step function, centered at zero.
def stepFunction(sum):
    if sum <= 0:
        return 0
    else:
        return 1

# Function to implement a sigmoiod, centered at zero.
def sigmoidFunction(sum):
    return 1.0/(1 + math.exp(sum * -1))
            
# Load the data
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
weights = [random.random(), random.random(), random.random()]
errorRates = []

#
# Train the perceptron
#
lastErrorRate = 1.0
for i in range(10000):
    # Generalised Delta rule for one epoch.
    #
    # For this one we are trying to distinguish 0/1 (class 0) from 2 (class 1)
    errors = 0
    error = 0
    for i in range(len(X)):
        sum = weightedSum(weights, [X[i][0], X[i][1]])
        
        # Real and desired outputs
        g_s = sigmoidFunction(sum)

        # Class 0 is setosa and easiest to classify, class 2 is
        # virginia and also makes sense to try to classify with a
        # single perceptron.
        if y[i] == 2:
            target = 1
        else:
            target = 0

        error = error + abs(target - g_s)
        
        if g_s > 0.5:
            output = 1
        else:
            output = 0

        if output != target:
            errors = errors + 1
            
        # Update weights.
        #
        # This is a bit of a nasty hack. A clean solution would treat
        # all the weights the same and have a single update. But that
        # means modifying X.
        weights[0] = weights[0] + alpha * (target - g_s) * g_s * (1 - g_s)
        
        for j in [0, 1]:
            weights[j+1] = weights[j+1] + alpha * (target - g_s) * g_s * (1 - g_s) * X[i][j]

    # Not currently using this error rate data
    errorRate = float(errors)/len(X)
    errorRates.append(errorRate)

print(errorRate, error)
print(weights)

#
# Now run the trained perceptron on the data:
#
z = []
for i in range(len(X)):
    sum = weightedSum(weights, [X[i][0], X[i][1]])
        
    # Real and desired outputs
    g_s = sigmoidFunction(sum)
    
    if g_s <= 0.5:
        output = 0
    else:
        output = 1
        
    z.append(output)
        
#
# Plot the results
#
# Plot the results of the trained perceptron and the original data.
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
