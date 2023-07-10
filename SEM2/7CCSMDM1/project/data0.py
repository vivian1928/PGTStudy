
# data0.py
# generates synthetic data for clustering
# @author: letsios, sklar
# @created: 12 Jan 2021

import sklearn.datasets as data
import matplotlib.pyplot as plt

DEBUGGING = True
PLOTS_DIR = '../plots/'


# generate synthetic data for clustering
STD = 1
samples = 1000
# X, clusters = data.samples_generator.make_blobs( n_samples=1000, n_features=2, cluster_std=STD )
(X, clusters) = data.make_blobs( n_samples=samples, n_features=2, cluster_std=STD )
# set number of instances
M = len( X )

# (optionally) print some info about the data set
if DEBUGGING:
    print( 'number of instances = ' + str( M ))

# plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
plt.plot( X[:,0], X[:,1], 'g.', markersize=10 )
# save plot
plt.savefig( PLOTS_DIR + 'data0-' + str(STD) + '-raw.png' )
plt.show()
plt.close()
