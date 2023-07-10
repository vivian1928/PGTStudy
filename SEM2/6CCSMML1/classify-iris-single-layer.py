# classify-iris-single-layer.py
# parsons/23-feb-2019
#
# Using a single layer neural network (just two perceptrons) on the
# Iris dataset.
#
# Code is based on:
#
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import math
import random
import numpy as np
import matplotlib.pyplot as plt
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

# Function to implement a sigmoid, centered at zero.
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
# Here we build a two unit single-layer network and train both units
# independently, but together.
#

# There are two inputs, so three weights (one per input and one for the bias).
weights1 = [random.random(), random.random(), random.random()]
weights2 = [random.random(), random.random(), random.random()]

#
# Train the perceptrons
#
errorRates = []
for i in range(10000):
    # Generalised Delta rule for one epoch.
    #
    # For this one we are trying to distinguish 0/1 (class 0) from 2 (class 1)
    errors = 0
    error = 0
    for i in range(len(X)):
        sum1 = weightedSum(weights1, [X[i][0], X[i][1]])
        sum2 = weightedSum(weights2, [X[i][0], X[i][1]])
        
        # Real and desired outputs
        g_s1 = sigmoidFunction(sum1)
        g_s2 = sigmoidFunction(sum2)

        # Class 0 is setosa and easiest to classify, class 2 is
        # virginia and also makes sense to try to classify with a single perceptron.
        if y[i] == 0:
            target1 = 1
        else:
            target1 = 0
        if y[i] == 2:
            target2 = 1
        else:
            target2 = 0

        error = error + abs(target1 - g_s1) + abs(target2 - g_s2)
        
        if g_s1 > 0.5:
            output1 = 1
        else:
            output1 = 0
        if g_s2 > 0.5:
            output2 = 1
        else:
            output2 = 0
            
        if output1 != target1 or output2 != target2:
            errors = errors + 1
            
        # Update weights.
        #
        # This is a bit of a nasty hack. A clean solution would treat
        # all the weights the same and have a single update. But that
        # means modifying X.
        weights1[0] = weights1[0] + alpha * (target1 - g_s1) * g_s1 * (1 - g_s1)
        weights2[0] = weights2[0] + alpha * (target2 - g_s2) * g_s2 * (1 - g_s2)
         
        for j in [0, 1]:
            weights1[j+1] = weights1[j+1] + alpha * (target1 - g_s1) * g_s1 * (1 - g_s1) * X[i][j]
            weights2[j+1] = weights2[j+1] + alpha * (target2 - g_s2) * g_s2 * (1 - g_s2) * X[i][j]

    # We collect the error rate over time but aren't doing anything
    # with this right now.
    errorRate = float(errors)/len(X)
    errorRates.append(errorRate)

print(errorRate, error)
print(weights1, weights2)

#
# Now run the perceptron on the data:
#
z = []
for i in range(len(X)):
    sum1 = weightedSum(weights1, [X[i][0], X[i][1]])
    sum2 = weightedSum(weights2, [X[i][0], X[i][1]])
    g_s1 = sigmoidFunction(sum1)
    g_s2 = sigmoidFunction(sum2)

    # unit 1 is trained to recognise class 0 when g_s1 > 0.5 and
    # unit 2 is trained to recognise class 2 when g_s2 > 0.5, so
    # we can combine thse to get all three classes:
    if g_s1 > 0.5:
        output = 0
    elif g_s2 > 0.5:
        output = 2
    else:
        output = 1
            
    z.append(output)

#
# Plot the results.
#
# Plot the results of the single-layer network
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30, c=z, cmap='autumn')
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Also plot the original data
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float), cmap='autumn')
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()
