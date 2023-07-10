#--
# km1.py
# runs kmeans clustering using synthetic data
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import numpy as np
import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics


DEBUGGING = False
PLOTS_DIR = '../plots/'

# define markers for up to 10 clusters
CLUSTER_MARKERS = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]


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
plt.savefig( PLOTS_DIR + 'km1-data.png' )
plt.show()
plt.close()

#-loop through various numbers of clusters
BC = np.zeros( 10 ) # between cluster
WC = np.zeros( 10 ) # within cluster
IN = np.zeros( 10 ) # inertia (within cluster)
SC = np.zeros( 10 ) # silhouette coefficient
CH = np.zeros( 10 ) # calinksi-harabasz
for K in range( 2, 10 ):
    if DEBUGGING:
        print('NUMBER OF CLUSTERS = ', K)
    km = cluster.KMeans( n_clusters=K )
    km.fit( X )
    if DEBUGGING:
        print ('cluster centres:')
        for k in range( K ):
            print('c{} = [{} {}]'.format( k, km.cluster_centers_[k][0], km.cluster_centers_[k][1] ))
            #print 'clusters:'
            #for j in range( M ):
            #    print '[%0.f %0.f] -> c%0d' % ( X[j][0], X[j][1], km.labels_[j] )
        print()
        print('within-cluster score = ' + str( km.inertia_ ))
    # record inertia score (within clusters, computed by scikit function)
    IN[K] = km.inertia_

    #-compute silhouette score
    SC[K] = metrics.silhouette_score( X, km.labels_, metric='euclidean' )

    #-compute calinski-harabasz score
    CH[K] = metrics.calinski_harabasz_score( X, km.labels_ )

    #-tally members of each cluster
    members = [[] for i in range( K )] # lists of members of each cluster
    for j in range( M ): # loop through instances
        members[ km.labels_[j] ].append( j ) # add this instance to cluster returned by scikit function

    #-compute the within-cluster score
    within = np.zeros(( K ))
    for i in range( K ): # loop through all clusters
        within[i] = 0.0
        for j in members[i]: # loop through members of this cluster
            # tally the distance to this cluster centre from each of its members
            within[i] += ( np.square( X[j,0]-km.cluster_centers_[i][0] ) + np.square( X[j,1]-km.cluster_centers_[i][1] ))
    WC[K] = np.sum( within )

    #-compute the between-cluster score
    between = np.zeros(( K ))
    for i in range( K ): # loop through all clusters
        between[i] = 0.0
        for l in range( i+1, K ): # loop through remaining clusters
            # tally the distance from this cluster centre to the centres of the remaining clusters
            between[i] += ( np.square( km.cluster_centers_[i][0]-km.cluster_centers_[l][0] ) + np.square( km.cluster_centers_[i][1]-km.cluster_centers_[l][1] ))
    BC[K] = np.sum( between )

    #-compute overall clustering score
    score = BC[K] / WC[K]

    #-print results for this value of K
    print('K={}  WC={}  BC={}  score={}  inertia={}  silhouette={}  calinski-harabasz={}'.format( K, WC[K], BC[K], score, IN[K], SC[K], CH[K] ))

    #-plot clusters for this value of K
    plt.figure()
    for j in range( M ):
        plt.plot( X[j][0], X[j][1], CLUSTER_MARKERS[km.labels_[j]] )
    plt.xlabel( 'X[0]' )
    plt.ylabel( 'X[1]' )
    plt.title( 'k-means clustering, K=' + str( K ))
    plt.savefig( PLOTS_DIR + 'km1-' + str( K ) + '.png' )
    #plt.show()
    plt.close()

#-plot overall scores
plt.figure()
plt.plot( BC, linewidth=2, label='Between Cluster' )
plt.plot( WC, linewidth=2, label='Within Cluster' )
plt.plot( SC*1000, linewidth=2, label='Silhouette Coefficient(x1K)' )
plt.plot( CH, linewidth=2, label='Calinski-Harabasz' )
plt.xlabel( 'K' )
plt.legend( loc='best' )
plt.savefig( PLOTS_DIR + 'km1-scores.png' )
plt.show()
plt.close()
