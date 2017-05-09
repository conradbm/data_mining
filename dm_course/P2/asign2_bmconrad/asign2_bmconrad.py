from __future__ import division

"""
# File Name: asign2_bmconrad.py
# Author: Blake Conrad
# Content: HW2 Kmeans Clustering
"""

"""
Import necessary libraries
"""
import csv
import sys
from random import randint
import numpy as np
import scipy.spatial.distance as dist

"""
Function: head
Arguments: numpy.ndarry
Return: void
Objective: Print first 5 obervations of argument given
"""
def head(D):
    if(len(D) >= 5):
        print D[:5]
    else:
        print D[:len(D)]
    

"""
Function: read_input_file
Arguments: string, string
Return: list()
Objective: given a filename and a tuning, read the file and return a list of strings
"""
def read_input_file(fileName, tuning):
    
    rowContainer=list()
    with open(fileName, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rowContainer.append(row)

    return rowContainer



"""
Function: getCentroids
Arguments: int, numpy.ndarray, list (optional)
Return: numpy.ndarray -- k points each as a mean cluster centroid point
Objective: return the means array based on user input
"""
def getCentroids(k, D, *centroids):

    # k rows 4 columns
    means = np.empty((k,4), dtype=np.float)

    # If centroids are given    
    if centroids:
        counter=0
        for item in centroids[0]:
            try:
                means[counter] = D[int(item[0])]
                
            except IndexError:
                print "The index, ", counter," couldn't be found. Choosing a random one instead."
                print "This program accounts for 1 extra newline character at the end of the file. It is possible you had many."
                print centroids[0]
                means[counter] = D[randint(0,len(D)-1)]
            counter = counter + 1
                      
    # If centroids are not given
    else:
        for i in range(k):
            means[i] = D[randint(0,len(D)-1)]

    # Return the mean candidates
    return means


"""
Function: find_min_idx
Arguments: numpy.array, numpy.ndarray
Return: int -- cluster ID
Objective: find the closest cluster mean index to a point xj
"""
def find_min_idx(xj, M):
    min_mean_dist = 9999999.00
    idx=0
    counter=0
    options=list()
    for Mj in M:
        tmp = np.linalg.norm(xj - Mj)
 #       tmp2 = np.sqrt(sum([ np.power((xj - row),2) for row in M]))

        options.append(tmp)
        if(tmp < min_mean_dist):
            min_mean_dist = tmp
            idx=counter
        counter=counter+1


    #print "Smallest idx: ", idx
    #print options
    #print xj
    #raw_input("~~~")
    return idx

"""
Function: Kmeans
Arguments: numpy.ndarray,int,float, list (optional)
Return: list() -- each points cluster assignment in D
Objective: Cluster the data into K groups, returning the index of each points association to a group.
"""
def Kmeans(D, k, e, *centroids):

    # Data containers and ticker variables
    t=0
    meanList = list()
    finalClusterAssignments = list()
    distList = list()
    
    # Get initial centroids (e.g., if K==5, len(means)==5)
    means = getCentroids(k, D, *centroids)
    
    # To confirm D and means are as expected
    #print "Head D" 
    #head(D)
    #print "Head means"
    #head(means)

    # Begin with meanList[0] = means
    meanList.append(means)
    
    while(True):

        # Assign each points cluster index (i.e., len(Cj) == number of data points)
        Cj = list()
        for xj in D:            
            val = find_min_idx(xj, meanList[t])
            Cj.append(val)

        # To confirm clusters are updating
        #print "head(Cj)"
        #print Cj[:15]
        #print
        #print "tail(Cj)"
        #print Cj[-15:]
        #print
        #raw_input("...")
        
        # Update containers and ticker variables
        
        meanList.append(np.array(means, dtype=np.float))
        listOfRows = list()
        t=t+1
        for i in range(k):
            for j in range(len(D)):
                if(Cj[j] == i):
                    listOfRows.append(D[j])
            if(len(listOfRows) != 0):
                meanList[t][i] = sum(listOfRows)/max(len(listOfRows),1)
            listOfRows = list()
        
        # To confirm means are updating
        #print "Old means:"
        #print meanList[t-1]
        #print
        #print "Updated Means:"
        #print meanList[t]
        #print

        # Get distance from new mean to old mean.
        dists=list()
        for i in range(k):
            d = np.linalg.norm(meanList[t][i] - meanList[t-1][i])
            dists.append(d)

        val = sum(dists)
        
        # To confirm distances make sense
        #print "Distances: " ,dists
        #print
        
        if (val <= e):
            finalClusterAssignments = Cj
            break
        else:
            continue

    clusterCount = getClusterCounts(finalClusterAssignments, D, k)
    the_sse = getSSE(D,k, finalClusterAssignments, meanList[-1])
    
    
    print "Converged in ", t, " iterations.\n"
    print "Final means: ", meanList[-1], ""
    print "Final cluster assignments: (",len(finalClusterAssignments),"):\n", finalClusterAssignments
    print "Final size of each cluster: ", clusterCount
    print "Final SSE: ", the_sse

    # Only print purity if it makes sense, i.e., k=3 because we know those predefined values
    if(k==3):
        the_purity = getPurity(Cj)
        print "Purity: ", the_purity
    return finalClusterAssignments, meanList[-1]


"""
Function: getPurity
Arguments: list
Return: Based on defined input, return the purity of a single Kmeans run.
"""
def getPurity(Cj):

    # Get the number of 0's, 1's, and 2's in the first 50 observations
    tmp0=0
    tmp1=0
    tmp2=0
    for i in range(50):
        if(Cj[i] == 0):
            tmp0 += 1
        if(Cj[i] == 1):
            tmp1 += 1
        if(Cj[i] == 2):
            tmp2 += 1
        
    T1 = np.array(list([tmp0, tmp1, tmp2]), dtype=np.float)

    # Get the number of 0's, 1's, and 2's in the second 50 observation
    tmp0=0
    tmp1=0
    tmp2=0
    for i in range(50,100):
        if(Cj[i] == 0):
            tmp0 += 1
        if(Cj[i] == 1):
            tmp1 += 1
        if(Cj[i] == 2):
            tmp2 += 1
    T2 = np.array(list([tmp0, tmp1, tmp2]), dtype=np.float)

    # Get the number of 0's, 1's, and 2's in the third 50 observation 
    tmp0=0
    tmp1=0
    tmp2=0
    for i in range(100,150):
        if(Cj[i] == 0):
            tmp0 += 1
        if(Cj[i] == 1):
            tmp1 += 1
        if(Cj[i] == 2):
            tmp2 += 1
    T3 = np.array(list([tmp0, tmp1, tmp2]), dtype=np.float)
        
    mainMatrix = np.matrix([T1,T2,T3], dtype=np.float)
    
    #purity_i = maxColumnValue / #true values for those 50
    purity_1 = np.divide(np.amax(mainMatrix[0]),50)       
    purity_2 = np.divide(np.amax(mainMatrix[1]),50)
    purity_3 = np.divide(np.amax(mainMatrix[2]),50)

    #ni/n = 50/150 = for index i, 50 were true out of 150 total observations
    ni_over_n = 1/3

    #get each Total Purity = {sum from i->k}(ni/n * purity_i)
    purities = list([purity_1*ni_over_n, purity_2*ni_over_n, purity_3*ni_over_n])
    return sum(purities)

    
"""
Function: SSE
Arguments: Data matrix, k, cluster assignments columnvector, meanMatrix
return: SSE float value
Objective: return the SSE of each point to its mean
"""
def getSSE(D, k, Cj, means):

    #SSE= sum i->k (||every point in each cluster - that cluster mean||)^2
    listOfHits=list()
    for i in range(k):
        for j in range(len(D)):
            if(Cj[j] == i):
                tmp = np.power(np.linalg.norm(D[j] - means[i]),2)
                listOfHits.append(tmp)
    sse = sum(listOfHits)
    return sse

"""
Function: getClusterCounts
Arguments: clusterassingments column vector, data matrix D, k
return: list of counts per cluster
Objective: return the SSE of each point to its mean
"""
def getClusterCounts(Cj, D, k):
    # Get size of clusters
    clusterCount = list()
    count=0
    for i in range(k):
        for j in range(len(D)):
            if(Cj[j] == i):
                count = count + 1
        clusterCount.append(count)
        count = 0
    return clusterCount


"""
Function: load_kmeans_no_centroids
Arguments: void
Return: void
Objective: Launch Kmeans given random centroids
"""
def load_kmeans_no_centroids():
    
        # Store Critical Variables
        dataFileName = sys.argv[1]
        k = int(sys.argv[2])
        e = 0.001
        
        # Input df
        D = read_input_file(dataFileName, "csv")
        D = np.matrix(D, dtype=np.float)

        # Specs of df
        rows,cols = D.shape
        
        # Standard Welcome Output
        print "File Name: ", dataFileName
        print "Data points: ", rows
        print "Dimension of each point: ", cols
        print "K specified as: ", k
        print "Centroids Specification: Random"
        
        # Apply K-means
        idx, centroids = Kmeans(D, k, e)


"""
Function: load_kmeans_given_centroids
Arguments: void
Return: void
Objective: Launch Kmeans given indexed centroids from data
"""
def load_kmeans_given_centroids():

        # Store Critical Variables
        dataFileName = sys.argv[1]
        k = int(sys.argv[2])
        centroidsFileName = sys.argv[3]
        e = 0.001

        # Input df
        D = read_input_file(dataFileName, "csv")
        D = np.matrix(D, dtype=np.float)

        # Specs of df
        rows,cols = D.shape
        
        # Standard Welcome Output
        print "File Name: ", dataFileName
        print "Data points: ", rows
        print "Dimension of each point: ", cols
        print "K specified as: ", k
        print "Centroids Specification: ", centroidsFileName
        
        # PRO-TIP: use the id's of the given values as your centroid points
        centroidsList = read_input_file(centroidsFileName, "linebyline")
        
        if(centroidsList[-1] == []):
            # new line issue as the last line of the file
            centroidsList = centroidsList[:-1]

        if(len(centroidsList) != k):
            print "Warning: Number of lines in ", centroidsFileName, " and given K do not match."
            print "Setting K =",len(centroidsList),"\n"
            k=len(centroidsList)

        # Apply K-means
        idx, centroids = Kmeans(D, k, e, centroidsList)
        
"""
Function: main
Arguments:
    - sys.argv[0] program name
    - sys.argv[1] specified k
    - sys.argv[2] initial centroids (OPTIONAL)
Return: void
Objective: 
    1. The number of data points in the input file, the dimension, and the value of k
    2. The number of iterations the program took for convergence
    3. The final mean of each cluster and the SSE score (sum of square error)
    4. The final cluster assignment of all the points.
    5. The final size of each cluster.
"""
def main():

    print
    
    # Default Centroids
    if(len(sys.argv) == 3):
        load_kmeans_no_centroids()
        
    # Selected centroids
    elif(len(sys.argv) == 4):
        load_kmeans_given_centroids()

if __name__ == '__main__':
  main()
