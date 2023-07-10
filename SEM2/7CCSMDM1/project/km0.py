#--
# km0.py
# perform kmeans clustering manually, using synthetic data
# @author: letsios, sklar
# @created: 12 Jan 2021
#
#--

import random
import numpy as np
import sklearn.datasets as data
import matplotlib.pyplot as plt

DEBUGGING = False
PLOTTING = True
PLOTS_DIR = '../plots/'

# define markers for up to 10 clusters
CLUSTER_MARKERS = [ 'bo', 'rv', 'c^', 'm<', 'y>', 'ks', 'bp', 'r*', 'cD', 'mP' ]


#-generate synthetic data for clustering
X, clusters = data.make_blobs( n_samples=100, n_features=2, cluster_std=2, random_state=2021 )
# save number of instances, as well as minimum and maximum values in the data set
M = len( X )
minX0 = np.min( X[:,0] )
maxX0 = np.max( X[:,0] )
minX1 = np.min( X[:,1] )
maxX1 = np.max( X[:,1] )

#-(optionally) print some info about the data set
if DEBUGGING:
    print('number of instances = ' + str( M ))
    print('X[:,0] min={} max={}   X[:,1] min={} max={}'.format( minX0, maxX0, minX1, maxX1 ))

#-plot raw data --- always a good idea to do this!
if PLOTTING:
    plt.figure()
    # plot data points
    plt.plot( X[:,0], X[:,1], 'g.', markersize=10 )
    # save plot
    plt.savefig( PLOTS_DIR + 'my-kmeans-data.png' )
    plt.show()
    plt.close()

#-loop through various numbers of clusters
BC = np.zeros( 10 )
WC = np.zeros( 10 )
for K in range( 2, 10 ):

    #-initialise clusters by finding random points for K cluster centres,
    # using the minimum and maximum values in the data set
    centres = np.zeros(( K, 2 ))
    for i in range( K ):
        centres[i,0] = random.uniform( minX0, maxX0 )
        centres[i,1] = random.uniform( minX1, maxX1 )
    if DEBUGGING:
        print('initial cluster centres=')
        for i in range( K ):
            # print '%d=(%f,%f) ' % ( i, centres[i,0], centres[i,1] ),
            print('{}=({},{}) '.format( i, centres[i,0], centres[i,1] ))
        print()

    #-compute distance from each point in the data set to each cluster
    # center, using the Euclidean distance
    dist    = np.zeros(( M, K )) # for each instance, distance to each cluster centre
    labels  = np.zeros( M, dtype=int ) # for each instance, label of cluster with closest centre
    members = [[] for i in range( K )] # lists of members of each cluster
    converged = False
    iters = 0
    while ( not converged ):
        if DEBUGGING:
            print('iteration: ', iters)
        converged = True
        for j in range( M ):
            if DEBUGGING:
                print('distances from ({},{}): '.format( X[j,0], X[j,1] ))
            for i in range( K ):
                dist[j,i] = np.sqrt( np.square( X[j,0]-centres[i,0] ) + np.square( X[j,1]-centres[i,1] ))
                if DEBUGGING:
                    print('( {}, {} )'.format( i, dist[j,i] ))
            this_cluster = np.argmin( dist[j,:] )
            if ( labels[j] != this_cluster ):
                converged = False
            labels[j] = this_cluster
            members[this_cluster].append( j )
        #-plot clusters
        if PLOTTING:
            plt.figure()
            for j in range( M ):
                plt.plot( X[j,0], X[j,1], CLUSTER_MARKERS[labels[j]] )
            plt.xlabel( 'X[0]' )
            plt.ylabel( 'X[1]' )
            plt.savefig( PLOTS_DIR + 'my-kmeans-K' + str( K ) + '-iter' + str( iters ) + '.png' )
            #plt.show()
            plt.close()
        #-compute new cluster centres
        for i in range( K ):
            if DEBUGGING:
                print('cluster {}, size={}'.format( i, len( members[i] )))
            if ( len( members[i] ) > 0 ):
                centres[i,0] = 0
                centres[i,1] = 0
                for m in range( len( members[i] )):
                    mx = members[i][m]
                    centres[i,0] += X[mx,0]
                    centres[i,1] += X[mx,1]
                centres[i,0] /= len( members[i] )
                centres[i,1] /= len( members[i] )
            else:
                centres[i,0] = random.uniform( minX0, maxX0 )
                centres[i,1] = random.uniform( minX1, maxX1 )
        #-iterate again...
        # (convergence--stopping condition--is checked at top of while loop)
        iters += 1

    #-compute the within-cluster score
    within = np.zeros(( K ))
    for i in range( K ): # loop through all clusters
        within[i] = 0.0
        for j in members[i]: # loop through members of this cluster
            # tally the distance to this cluster centre from each of its members
            within[i] += ( np.square( X[j,0]-centres[i,0] ) + np.square( X[j,1]-centres[i,1] ))
    WC[K] = np.sum( within )

    #-compute the between-cluster score
    between = np.zeros(( K ))
    for i in range( K ): # loop through all clusters
        between[i] = 0.0
        for l in range( i+1, K ): # loop through remaining clusters
            # tally the distance from this cluster centre to the centres of the remaining clusters
            between[i] += ( np.square( centres[i,0]-centres[l,0] ) + np.square( centres[i,1]-centres[l,1] ))
    BC[K] = np.sum( between )

    #-compute overall clustering score
    score = BC[K] / WC[K]

    #-print results for this value of K
    print('K={}  WC={}  BC={}  score={}'.format( K, WC[K], BC[K], score ))


#-plot overall scores
plt.figure()
plt.plot( BC, linewidth=2, label='Between Cluster' )
plt.plot( WC, linewidth=2, label='Within Cluster' )
plt.xlabel( 'K' )
plt.legend( loc='best' )
plt.savefig(PLOTS_DIR + 'my-kmeans-bc-wc.png' )
plt.show()
plt.close()
