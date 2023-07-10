#--
# tree0.py
# Builds a Decision Tree using the Iris data set
# @author: letsio, sklar
# @created: 12 Jan 2021
#
#--

import sklearn.datasets as data
import sklearn.model_selection as model_select
import sklearn.tree as tree
import sklearn.metrics as metrics

DOT_FILE = '../plots/iris-tree.dot'
DEBUGGING = True

# load the built-in iris data set
iris = data.load_iris()
if ( DEBUGGING ):
    print('classes = ', iris.target_names)
    print('attributes = ', iris.feature_names)

# split the data into training and test sets
X_train, X_test, y_train, y_test = model_select.train_test_split( iris.data, iris.target, random_state=0 )
M_train = len( X_train )
M_test = len( X_test )
if ( DEBUGGING ):
    print('number of training instances = ' + str( M_train ))
    print('number of test instances = ' + str( M_test ))

# initialise the decision tree
clf = tree.DecisionTreeClassifier( random_state = 0 )

# fit the tree model to the training data
clf.fit( X_train, y_train )

# predict the labels for the test set
y_hat = clf.predict( X_test )
# count the number of correctly predicted labels
count = 0.0
for i in range( M_test ):
    if ( y_hat[i] == y_test[i] ):
        count += 1
score = ( count / M_test )
print('number of correct predictions = {} out of {} = {}'.format( count, M_test, score ))

print('training score = ', clf.score( X_train, y_train ))
print('test score = ', clf.score( X_test, y_test ))
print('accuracy score = ', metrics.accuracy_score( y_test, y_hat ))

cm = metrics.confusion_matrix( y_test, y_hat )
print('confusion matrix =')
#print '%10s\t%s' % ( ' ','predicted-->' )
print('\t predicted-->')
#print '%10s\t' % ( 'actual:' ),
print('actual:', end='')
for i in range( len( iris.target_names )):
    #print '%10s\t' % ( iris.target_names[i] ),
    print( iris.target_names[i], end='' )
#print '\n',
print()
for i in range( len( iris.target_names )):
    #print '%10s\t' % ( iris.target_names[i] ),
    for j in range( len( iris.target_names )):
        #print '%10s\t' % ( cm[i,j] ),
        print(cm[i,j], end='') 
    #print '\n',
    print()
# print '\n',
print()

print('precision score = tp / (tp + fp) =')
precision = metrics.precision_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    print('\t {} = {}'.format( iris.target_names[i], precision[i] ))

print('recall score = tp / (tp + fn) =')
recall = metrics.recall_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    print('\t {} = {}'.format( iris.target_names[i], recall[i] ))

print('f1 score = 2 * (precision * recall) / (precision + recall) =')
f1 = metrics.f1_score( y_test, y_hat, average=None )
for i in range( len( iris.target_names )):
    print('\t {} = {}'.format( iris.target_names[i], f1[i] ))

# what does the tree look like?
print('decision path: ')
print(clf.decision_path( iris.data ))                                                                                

# output the tree to "dot" format for later visualising
tree.export_graphviz( clf, out_file = DOT_FILE, class_names=iris.target_names, impurity=True )
print('output dot file written to: ', DOT_FILE)

# then run "dot" from the unix command line to generate an image file from the dot file:
#  unix-prompt$ dot iris-tree.dot -Tpng >iris-tree.png
# or run "graphviz"
