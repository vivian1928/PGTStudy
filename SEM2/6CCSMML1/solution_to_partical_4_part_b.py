#--
# solution_to_partical_4_part_b.py
# ganzer-ripoll/25-feb-2017
#
# This code is based the k-means code by parsons+martinez-miranda in the file
# solution_to_practical_5.py
#
# The input data is the processed Cleveland data from the "Heart
# Diesease" dataset at the UCI Machine Learning repository:
#
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.
#--

import sys
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


# --------- K-means redux --------------

print("\n\n\t\t ---- K-mean redux ----  ")

#
# Define constants
#

datafilename = 'StoneFlakes.dat' # input filename
lbi      = 0                     # column indexes in input file
rti      = 1
wdi      = 2
fla      = 3
psf      = 4
fsf      = 5
zdf1     = 6
prozd    = 7

#  Attributes:

# LBI: Length-breadth index of the striking platform 
# RTI: Relative-thickness index of the striking platform 
# WDI: Width-depth index of the striking platform 
# FLA: Flaking angle (the angle between the striking platform and the splitting surface) 
# PSF: platform primery (yes/no, relative frequency) 
# FSF: Platform facetted (yes/no, relative frequency) 
# ZDF1: Dorsal surface totally worked (yes/no, relative frequency) 
# PROZD: Proportion of worked dorsal surface (continuous)

# Since feature names are not in the data file, we code them here
feature_names = ['lbi','rti','wdi','fla','psf','fsf','zdf1','prozd' ]

num_samples = 79 # size of the data file. 
num_features = 8

#
# Open and read data file in csv format
#
# After processing:
# 
# data   is the variable holding the features;

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features ))
        i = 0
        for j, d in enumerate( indata ):
            if j==0: continue       # Avoid the first line of the file because it has
            ok = True               # column identifiers in it.
            # Delete the ID string of every line
            save=""
            for k in d[0]:
                try:
                    save=save+str(int(k))
                except Exception as e:
                    if k=='.':
                        save=save+'.'
                    if k=='?':
                        save=save+'?'
            d[0]=save
            # d[0] = 0
            for k in range(1,num_features):     # If a feature has a missing value
                if '?' in d[k]:                 # we don't use that record.
                    ok = False
            if ( ok ):
                data[i] = np.asarray( d[:], dtype=np.float64 )
                i = i + 1
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

# How many records do we have?
num_samples = i
print("\n Number of samples:", num_samples)

data = data[:i]
print(data)

#
# A function to compute the Euclidian distance between two points with
# multidimensional coordinates.
#
def distance_multidimensional(point1, point2):
    sqSum=0.0
    for i in range(len(point1)):
        print(sqSum)
        sqSum += (point1[i] - point2[i]) ** 2
    return math.sqrt(sqSum)

minClusterNum=2
maxClusterNum=6

# A loop for each number of clusters

error = 0.01   # convergence is no cluster center moving more than this

for clusterNum in range (minClusterNum, maxClusterNum+1):

    print("\n"+str(clusterNum)+" clusters:\n")

    #
    # Pick inital cluster centres.
    #
    # Get max and min values of the features we cluster on
    X_min=[]
    X_max=[]
    for i in range(num_features):
        X_min.append(math.floor(data[:,i].min()))
        X_max.append(math.ceil(data[:,i].max()))
        
    # now pick random points in that space.
    clusters = []
    for i in range(clusterNum):
        center=[]
        for j in range(num_features):
            center.append(random.randrange(X_min[j],  X_max[j]))
        clusters.append(center)

    # print initial clsuter centres just in case some of them are duplicates
    print("Initial clusters:")
    print(clusters)
    print("\n")

    #
    # Now run K-means.
    #
    # First build our own version of the data, Xcluster, with an
    # additional element that identifies which cluster the point is
    # in. Initialise that cluster to 0.

    Xcluster = []
    for i in range(len(data)):
        Xcluster.append([data[i], 0])

    stop = 0

    while stop == 0:
        # Assign each point to the cluster with the nearest centre.
        for i in range(len(Xcluster)):
            distances = []
            # For each data point, compute the distance to each cluster center
            for j in range(len(clusters)):
                distances.append(distance_multidimensional(Xcluster[i][0], clusters[j]))

            # Then assign the point to the cluster with the nearest center.
            min_distance=min(distances)
            for j in range(len(clusters)):
                if min_distance == distances[j]:
                    Xcluster[i][1]=j


        # Recompute cluster centres.
        cluster_newCenters=[]
        for i in range(clusterNum):
            cluster_newCenters.append([0]*num_features)
        counts=[0]*clusterNum

        # Sum up x and y coordinate values of points in each cluster.
        for i in range(len(Xcluster)):
            for j in range(len(clusters)):
                if Xcluster[i][1] == j:
                    for k in range(num_features):
                        cluster_newCenters[j][k] += Xcluster[i][0][k]
                    counts[j] += 1
                    
        # Use that to compute new cluster centres and how much each centre
        # moves
        diffs = [0]*clusterNum
        for i in range(len(clusters)):
            if counts[i] > 0:
                for j in range(num_features):
                    cluster_newCenters[i][j] = cluster_newCenters[i][j]/counts[i]
                diffs[i] = distance_multidimensional(cluster_newCenters[i], clusters[i])
                clusters[i] = cluster_newCenters[i]
    
        # Stop when all centres move less than error

        stop = 1
        if max(diffs) > error:
            stop=0



    # End of while loop


    # Extract the cluster identifer into a separate list
    labels = []
    for i in range(len(Xcluster)):
        labels.append(Xcluster[i][1])

    # Compute the Silhouette score and the Calinski Harabaz index and print the results    

    sIndex=silhouette_score(data,labels)
    cIndex=calinski_harabasz_score(data,labels)

    print("\t silhouette score = "+str(sIndex))
    print("\t calinski harabaz score = "+str(cIndex))
    

    #--
    # plot results
    #--
    plt.figure()
    plt.rc( 'xtick', labelsize=8 )
    plt.rc( 'ytick', labelsize=8 )
    for x in range( 0, num_features ):
        for y in range( x+1,num_features ):
            plt.subplot( num_features, num_features, y*num_features+x+1 )
            plt.scatter( data[:,x], data[:,y], c=labels )
            plt.xlabel( feature_names[x], fontsize=8 )
            plt.ylabel( feature_names[y], fontsize=8 )
            plt.axis( "tight" )
    plt.show()

# End of for clusterNum loop






