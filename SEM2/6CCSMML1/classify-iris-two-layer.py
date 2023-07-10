# classify-iris-two-layer.py
# parsons/28-feb-2019
#
# Using hand-crafted, very simple neural network to classify the iris
# dataset.
#
# The network has one hidden layer of two units, and uses two output
# units.
#
# Code is based, in part, on:
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
alpha = 0.001

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
# Here we build a simple neural network, with two inputs, two hidden
# units and two outputs:
#
# I - H1 - 01
#   X    X
# I - H2 - 02
#
# Building the network really consists of initialising the weight
# vectors.

# There are four weight vectors: for the two hidden units and the
# output unit. Each has three weights, one per input and one for the
# bias).
weights_h1 = [random.random(), random.random(), random.random()]
weights_h2 = [random.random(), random.random(), random.random()]
weights_o1  = [random.random(), random.random(), random.random()]
weights_o2  = [random.random(), random.random(), random.random()]

#
# Train  the network
#
errorRates = []
# Trial and error led to this number of epochs given the learning rate.
for i in range(5000):
    #
    # Backpropagation for one epoch.
    #
    error = 0
    errors = 0
    for i in range(len(X)):
        # Step 1
        # We forward propagate to establish the output.
        #
        # Hidden units have the X values as inputs, output units have
        # hidden unit outputs as inputs.
        sum_h1 = weightedSum(weights_h1, [X[i][0], X[i][1]])
        g_sh1  = sigmoidFunction(sum_h1)
        sum_h2 = weightedSum(weights_h2, [X[i][0], X[i][1]])
        g_sh2  = sigmoidFunction(sum_h2)
        sum_o1  = weightedSum(weights_o1, [g_sh1, g_sh2])    
        g_so1   = sigmoidFunction(sum_o1)
        sum_o2  = weightedSum(weights_o2, [g_sh1, g_sh2])    
        g_so2   = sigmoidFunction(sum_o2)
        
        # Step 2
        # Determine the error at the output.
        #
        # Class 0 is setosa and easiest to classify, class 2 is
        # virginia and is also easy to classify.  We'll try to
        # classify so that class 0 gives us 1 on o1 and the others
        # give 0 and so that class 2 gives us 1 on o2 and the other
        # classes give us 0
        if y[i] == 0:
            target1 = 1
        else:
            target1 = 0
        if y[i] == 2:
            target2 = 1
        else:
            target2 = 0

        error_o1 = target1 - g_so1
        error_o2 = target2 - g_so2
            
        Delta1 = error_o1 * g_so1 * (1 - g_so1)
        Delta2 = error_o2 * g_so2 * (1 - g_so2)

        # This next part allows us to track errors through the training
        # process (but it isn't part of backpropagation).
        error = error + abs(target1 - g_so1) + abs(target2 - g_so2)
        
        if g_so1 > 0.5:
            output1 = 1
        else:
            output1 = 0
        if g_so2 > 0.5:
            output2 = 1
        else:
            output2 = 0

        if output1 != target1 or output2 != target2:
            errors = errors + 1
            
        # Step 3
        # Update weights of output units.
        #
        # Updates depend on the outputs of the relevant hidden unit.
        weights_o1[0] = weights_o1[0] + alpha * Delta1
        weights_o1[1] = weights_o1[1] + alpha * Delta1 * g_sh1 
        weights_o1[2] = weights_o1[2] + alpha * Delta1 * g_sh2 

        weights_o2[0] = weights_o2[0] + alpha * Delta2
        weights_o2[1] = weights_o2[1] + alpha * Delta2 * g_sh1 
        weights_o2[2] = weights_o2[2] + alpha * Delta2 * g_sh2 

        # Step 4
        # Determine the fraction of Delta that each hidden unit is
        # responsible for
        Delta_h11 = g_sh1 * (1 - g_sh1) * weights_o1[1] * Delta1
        Delta_h12 = g_sh1 * (1 - g_sh1) * weights_o2[1] * Delta2
        Delta_h21 = g_sh2 * (1 - g_sh2) * weights_o1[2] * Delta1
        Delta_h22 = g_sh2 * (1 - g_sh2) * weights_o2[2] * Delta2

        # Step 5
        # Update the weights of the hidden units.
        weights_h1[0] = weights_h1[0] + alpha * (Delta_h11 + Delta_h12)
        weights_h2[0] = weights_h2[0] + alpha * (Delta_h21 + Delta_h22)
         
        for j in [0, 1]:
            weights_h1[j+1] = weights_h1[j+1] + alpha * (Delta_h11 + Delta_h12) * X[i][j]
            weights_h2[j+1] = weights_h2[j+1] + alpha * (Delta_h21 + Delta_h22) * X[i][j]

    # Here we store errors, but we don't want to do this for large
    # numbers of iterations.
    # errorRates.append(error)

print(error)
#print errorRates

# 
# Evaluate the trained network
#
# Note that we use all the data that we trained on to ``test'' the
# final version. This is, of ocurse, not a proper test, just an
# illustration of how well the network works on the test data.
#
# Run the network on the data
z = []
for i in range(len(X)):
    sum_h1 = weightedSum(weights_h1, [X[i][0], X[i][1]])
    g_sh1  = sigmoidFunction(sum_h1)
    sum_h2 = weightedSum(weights_h2, [X[i][0], X[i][1]])
    g_sh2  = sigmoidFunction(sum_h2)
    sum_o1  = weightedSum(weights_o1, [g_sh1, g_sh2])    
    g_so1   = sigmoidFunction(sum_o1)
    sum_o2  = weightedSum(weights_o2, [g_sh1, g_sh2])    
    g_so2   = sigmoidFunction(sum_o2)
        
    if g_so1 > 0.5 and g_so2 < 0.5:
        out_class = 0
    elif g_so1 < 0.5 and g_so2 > 0.5:
        out_class = 2
    else:
        out_class = 1
        
    z.append(out_class)

# Compute and print the error-rate:
errors = 0
for i in range(len(X)):
    if z[i] != y[i]:
        errors += 1
print(float(errors)/len(X))
              
# Plot the results of the single-layer network
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=30, c=z, cmap='autumn')
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Also plot the original data
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float), cmap='autumn')
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# Show the plot
plt.show()
