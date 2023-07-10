# solution_lab2.py
#
# Examples of linear regression and classification
#
# Code is based on:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_boston, make_regression
from sklearn.model_selection import train_test_split




# The main parameters of make-regression are the number of samples, the number
# of features (how many dimensions the problem has), and the amount of noise.
X, y = make_regression(n_samples=100, n_features=1, noise = 2)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=0.2, random_state=0)



# -----  PREVIOUS CODE  ------

# Create a regression model
regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

# Data on how good the model is:
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Plot outputs
plt.subplot( 3, 2, 1 )
plt.title("Linear Regression")
plt.scatter(X_test, y_test,  color='black',s=5)                       # Test data
plt.scatter(X_train, y_train,  color='red',s=5)                       # Training data
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=1) # The line we learnt

plt.xticks(()) # Don't clutter the plot with values on the axes
plt.yticks(())



# -------------- SOLUTIONS ---------------

# --------------  Part 3  ----------------

print("\n \t\t---- Part 3 ----\n")

# 1. Implement gradient descent as described above.
# 2. Plot the error/learning curve.
# 3. Compute the mean squared error for your model on your test data.
# 4. Modify your model (especially learning rate and the number of times the training data is used) until it performs as well as the scikit-learn built in method.
# 5. Vary the noise in the dataset and see how well your learning method compares with the scikit-learn method.

# Now, let's learn our own regression model.

# Parameters
alpha = 0.001  # Learning rate
repeats = 200

# Initializing variables
w0 = 0
w1 = 0
errors = []
points = []


# --- Stochastic Gradient Descent Method
print("\n -- Stochastic Gradient Descent:\n")

for j in range(repeats):
    for i in range(len(X_train)):

        # compute error
        predict = w0 + (X_train[i] * w1)
        error = y_train[i] - predict
        errors.append(error)             # Stash the error in an array

        # Update weights
        w0 = w0 + (alpha * error)
        w1 = w1 + ((alpha * error) * X_train[i])


print("w0: %2f" % w0)
print("w1: %2f" % w1)


# Compute mean using the testing data:

predicts = []
mean_error = 0
for i in range(len(X_test)):
    predict = w0 + (X_test[i] * w1)
    error = (y_test[i] - predict) ** 2      # Computing squared error
    mean_error  += error
    predicts.append(predict)

mean_error = mean_error / len(X_test)       # Computing mean squared error

print("Mean squared error: %2f" % mean_error)


# Plot the examples with the predictions and the squared error evolution

for i in range(len(errors)):
    errors[i] = errors[i] ** 2
    points.append(i)

plt.subplot( 3, 2, 3 )
plt.title("Stochastic Gradient")
plt.scatter(X_test, y_test, color="black",s=5)
plt.scatter(X_train, y_train, color="red",s=5)
plt.plot(X_test, predicts, color='blue', linewidth=1) # The line we learnt

plt.subplot( 3, 2, 4 )
plt.title("Stoch. Grad. Error")
plt.plot(points, errors)


# Now, let's do the same with Batch Gradient Gescent

# Same Parameters
alpha = 0.001  # Learning rate
repeats = 200

# Reset variables
w0 = 0
w1 = 0
errors = []
points = []


# --- Batch Gradient Descent Method

print("\n -- Batch Gradient Descent:\n")

for j in range(repeats):
    error_sum =0
    squared_error_sum=0
    error_sum_x=0
    for i in range(len(X_train)):

        # compute error
        predict = w0 + (X_train[i] * w1)
        squared_error_sum = squared_error_sum + (y_train[i]-predict)**2     # Error to produce plot
        error_sum = error_sum + y_train[i] - predict                        # Error to update w0
        error_sum_x = error_sum_x + (y_train[i] - predict)*X_train[i]       # Error to update w1

    # Update weights and append error
    w0 = w0 + (alpha * error_sum)
    w1 = w1 + (alpha * error_sum_x)
    errors.append(squared_error_sum/len(X_train))           # Stash the error in an array

print("w0: %2f" % w0)
print("w1: %2f" % w1)


# Compute mean using the testing data:

predicts = []
mean_error = 0
for i in range(len(X_test)):
    predict = w0 + (X_test[i] * w1)
    error = (y_test[i] - predict) ** 2      # Computing squared error
    mean_error  += error
    predicts.append(predict)

mean_error = mean_error / len(X_test)       # Computing mean squared error

print("Mean squared error: %2f" % mean_error)


# Plot he examples with the predictions and the squared error evolution

for i in range(len(errors)):
    points.append(i)

plt.subplot( 3, 2, 5 )
plt.title("Batch Gradient")
plt.scatter(X_train, y_train, color="red",s=5)
plt.scatter(X_test, y_test, color="black",s=5)
plt.plot(X_test, predicts, color='blue', linewidth=1) # The line we learnt


plt.subplot( 3, 2, 6 )
plt.title("Batch Grad. Error")
plt.plot(points, errors)
plt.tight_layout()
plt.show()


# --------------  Part 4  ----------------

print("\n \t\t---- Part 4 ----\n")

# 1. Extend your gradient descent program to handle multiple features.
# 2. Again, test this against the performance of the scikit-learn builtin function using means squared error.


# Let's get a sample with 3 features

# Parameter number of feeatures
num_features=3

X, y = make_regression(n_samples=100, n_features=num_features, noise = 2)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=0.2, random_state=0)

# Create a regression model
regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

# Data on how good the model is:
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))


# Convert to list and add the new attribute X[i][0]=1 to easy the computation
X_train=X_train.tolist()
y_train=y_train.tolist()
X_test=X_test.tolist()
y_test=y_test.tolist()
for i in range(len(X_train)):
	X_train[i]=[1]+X_train[i]

for i in range(len(X_test)):
	X_test[i]=[1]+X_test[i]


# Same Parameters
alpha = 0.001  # Learning rate
repeats = 200

# Reset variables

weights=[0]*(num_features+1) 		# Now we have a list of weights
errors = []
points = []

# --- Multivariate Gradient Descent Method

print("\n -- Multivariate Gradient Descent:\n")

for j in range(repeats):
	for i in range(len(X_train)):
		predict=0
    	# prediction
		for k in range(len(weights)):
			predict += (X_train[i][k] * weights[k])
	
		# compute error
		error = y_train[i] - predict
		errors.append(error)             # Stash the error in an array

        # Update weights
		for k in range(len(weights)):
			weights[k] +=  ((alpha * error) * X_train[i][k])
     

for k in range(len(weights)):
	print ("w"+str(k)+": %2f" % weights[k])


# Compute mean using the testing data:

predicts = []
mean_error = 0
for i in range(len(X_test)):
	
	#Prediction
	predict=0
	for k in range(len(weights)):
		predict +=  (X_test[i][k] * weights[k])
	predicts.append(predict)
	
	#Error
	error = (y_test[i] - predict) ** 2      # Computing squared error
	mean_error += error
	

mean_error = mean_error / len(X_test)       # Computing mean squared error

print("Mean squared error: %2f" % mean_error)


#Plot of the error evolution

for i in range(len(errors)):
	errors[i] = errors[i]**2
	points.append(i)


plt.title("Multivariate Gradient Descent Error")
plt.plot(points, errors)
plt.tight_layout()
plt.show()


# --------------  Part 5  ----------------

print("\n \t\t---- Part 5 ----\n")

# 1. Look at the Boston Housing dataset from scikit-learn. Import this using:

from sklearn.datasets import load_boston
boston = load_boston()

# This is not a very linear dataset, but if you pull out feature number 5:

X_train, X_test, y_train, y_test = train_test_split(boston.data[:, np.newaxis, 5], boston.target, test_size=0.2, random_state=0)

# you can find something that is reasonably linear (I get a mean squared error of around 46 and a variance score of 0.42).
# 2. Now do a multivariate regression on the Boston Housing data. Can you find a model that fits better?


# Create a regression model
regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

# Data on how good the model is:
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))



num_features=len(X_train[0])

# Convert to list and add the new attribute X[i][0]=1 to easy the computation

X_train=X_train.tolist()
y_train=y_train.tolist()
X_test=X_test.tolist()
y_test=y_test.tolist()
for i in range(len(X_train)):
	X_train[i]=[1]+X_train[i]

for i in range(len(X_test)):
	X_test[i]=[1]+X_test[i]


# Same Parameters
alpha = 0.001  # Learning rate
repeats = 1000

# Reset variables

weights=[0]*(num_features+1) 		# Now we have a list of weights
errors = []
points = []

# --- Multivariate Gradient Descent Method

print("\n -- Multivariate Gradient Descent Boston Data:\n")

for j in range(repeats):
	for i in range(len(X_train)):
		predict=0
    	# prediction
		for k in range(len(weights)):
			predict += (X_train[i][k] * weights[k])
	
		# compute error
		error = y_train[i] - predict
		errors.append(error)             # Stash the error in an array

        # Update weights
		for k in range(len(weights)):
			weights[k] +=  ((alpha * error) * X_train[i][k])
     

for k in range(len(weights)):
	print ("w"+str(k)+": %2f" % weights[k])


# Compute mean using the testing data:

predicts = []
mean_error = 0
for i in range(len(X_test)):
	
	#Prediction
	predict=0
	for k in range(len(weights)):
		predict +=  (X_test[i][k] * weights[k])
	predicts.append(predict)
	
	#Error
	error = (y_test[i] - predict) ** 2      # Computing squared error
	mean_error += error
	

mean_error = mean_error / len(X_test)       # Computing mean squared error

print("Mean squared error: %2f" % mean_error)


#Plot of the error evolution

for i in range(len(errors)):
	errors[i] = errors[i]**2
	points.append(i)


plt.title("Multivariate Gradient Descent Error")
plt.plot(points, errors)
plt.tight_layout()
plt.show()



