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
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import scipy.cluster.hierarchy as hierarchy



DEBUGGING = True
PLOTTING = True
PLOTS_DIR = '../plots/'

# define markers for up to 10 clusters
CLUSTER_MARKERS = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]


#-generate synthetic data for clustering
X, clusters = data.make_blobs( n_samples=1000, n_features=2, cluster_std=2, random_state=2019 )
# set number of instances
M = len( X )

#-(optionally) print some info about the data set
if DEBUGGING:
    print( 'number of instances = ' + str( M ))

#-plot raw data --- always a good idea to do this!
plt.figure()
# plot data points
plt.plot( X[:,0], X[:,1], 'g.', markersize=10 )
# save plot
plt.savefig( PLOTS_DIR + 'agg-data.png' )
#plt.show()
plt.close()

#-loop through various numbers of clusters
SC = np.zeros( 10 ) # silhouette coefficient
CH = np.zeros( 10 ) # calinksi-harabasz
for K in range( 2, 10 ):

    #-create clusters using sckit-learn's agglomerative clustering function
    ac = cluster.AgglomerativeClustering( n_clusters=K, linkage='average', affinity='euclidean' )
    ac.fit( X )

    #-compute silhouette score
    SC[K] = metrics.silhouette_score( X, ac.labels_, metric='euclidean' )

    #-compute calinski-harabasz score
    CH[K] = metrics.calinski_harabasz_score( X, ac.labels_ )

    #-tally members of each cluster
    members = [[] for i in range( K )] # lists of members of each cluster
    for j in range( M ): # loop through instances
        members[ ac.labels_[j] ].append( j ) # add this instance to cluster returned by scikit function

    #-print results for this value of K
    print('K={}  silhouette={}  calinski-harabasz={}'.format( K, SC[K], CH[K] ))

    #-plot dendrogram for this value of K
    if PLOTTING:
        plt.figure()
        # initialise data structure for scipy dendrogram printing function
        Z = np.empty( [ len( ac.children_ ), 4 ], dtype=float )
        # steps (A) and (B) below are thanks to:
        # https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
        # for hints on how to combine the sklearn agglomerative clustering with the dendrogram plotting function of scipy.
        # (A) compute distances between each pair of children: since we don't
        # have this information, we can use a uniform one for plotting
        cluster_distances = np.arange( ac.children_.shape[0] )
        # (B) compute the number of observations contained in each cluster level
        cluster_sizes = np.arange( 2, ac.children_.shape[0]+2 )
        for i in range( len( ac.children_ )):
            Z[i][0] = ac.children_[i][0]
            Z[i][1] = ac.children_[i][1]
            Z[i][2] = cluster_distances[i]
            Z[i][3] = cluster_sizes[i]
        # plot dendrogram
        hierarchy.dendrogram( Z )
        plt.savefig( PLOTS_DIR + 'agg-K' + str( K ) + '-dendrogram.png' )
        #plt.show()
        plt.close()

#-plot clusters for this value of K
    if PLOTTING:
        plt.figure()
        for j in range( M ):
            plt.plot( X[j][0], X[j][1], CLUSTER_MARKERS[ac.labels_[j]] )
        plt.xlabel( 'X[0]' )
        plt.ylabel( 'X[1]' )
        plt.title( 'agglomerative clustering, K=' + str( K ))
        plt.savefig( PLOTS_DIR + 'agg-K' + str( K ) + '.png' )
        #plt.show()
        plt.close()

#-plot overall scores
plt.figure()
plt.plot( SC*1000, linewidth=2, label='Silhouette Coefficient(x1K)' )
plt.plot( CH, linewidth=2, label='Calinski-Harabasz' )
plt.xlabel( 'K' )
plt.legend( loc='best' )
plt.savefig( PLOTS_DIR + 'agg-scores.png' )
plt.show()
plt.close()
