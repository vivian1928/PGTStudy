#--
# agg.py
# runs agglomerative (hierarchicial) clustering using synthetic data
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import numpy as np
import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.neighbors as neighbors
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import scipy.cluster.hierarchy as hierarchy


DEBUGGING = False
PLOTTING = True
PLOTS_DIR = '../plots/'


#-generate synthetic data for clustering
X, clusters = data.make_blobs( n_samples=1000, n_features=2, cluster_std=2, random_state=2019 )
# set number of instances
M = len( X )

#-(optionally) print some info about the data set
if DEBUGGING:
    print( 'number of instances = %d' % ( M ))

#-plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
plt.plot( X[:,0], X[:,1], 'g.', markersize=10 )
# save plot
plt.savefig( PLOTS_DIR + 'dbs-data.png' )
#plt.show()
plt.close()

#-examine the data by looking at the distances to the nearest
# neighbours for each instance in the data set. this will help us find
# good values for EPS.
nn = neighbors.NearestNeighbors( n_neighbors=2, metric='euclidean' )
nn.fit( X )
dist, ind = nn.kneighbors( X, n_neighbors=2 )
if PLOTTING:
    plt.figure()
    plt.plot( sorted( dist[:,1] ), linewidth=2 )
    plt.savefig( PLOTS_DIR + 'dbs-nn.png' )
    plt.show()
    plt.close()

#-try different values of EPS and MS
for ( EPS, MS ) in zip(( 0.5, 0.75, 0.75, 0.75 ), ( 1, 1, 5, 10 )):

    #-create clusters using sckit-learn's DBSCAN function
    db = cluster.DBSCAN( eps=EPS, min_samples=MS, metric='euclidean' )
    db.fit( X )

    # save number of clusters
    K = len( set( db.labels_ ))
    if DEBUGGING:
        print('number of clusters = ' + str( K ))
        print('clusters:')
        for j in range( M ):
            print('{} [{}  {}] -> {}'.format( j, X[j][0], X[j][1], db.labels_[j] ))
        # note: "Noisy samples are given the label -1."

    #-compute silhouette score
    SC = metrics.silhouette_score( X, db.labels_, metric='euclidean' )

    #-compute calinski-harabaz score
    CH = metrics.calinski_harabasz_score( X, db.labels_ )

    #-print results
    print('EPS={}  MS={}  K={}  silhouette={}  calinski-harabaz={}'.format( EPS, MS, K, SC, CH ))

    #-plot clusters
    if PLOTTING:
        color_list = plt.cm.Set1( np.linspace( 0, 1, K+1 ))
        plt.figure()
        plt.set_cmap( 'Set1' )
        for j in range( M ):
            plt.plot( X[j][0], X[j][1], marker='o', color=color_list[db.labels_[j]+1] )
        plt.xlabel( 'X[0]' )
        plt.ylabel( 'X[1]' )
        plt.title( 'DBSCAN clustering, MS=' + str( MS ))
        plt.savefig( PLOTS_DIR + 'dbs-MS' + str( MS ) + '-' + str( int( EPS*100 )) + '.png' )
        plt.show()
        plt.close()
