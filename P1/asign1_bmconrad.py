
# coding: utf-8

# - - - - - - - - - - - - - - - #
# Author: Blake Conrad
# Purpose: HW1 CSCI 48100
# - - - - - - - - - - - - - - - #

from __future__ import division
import sys
import numpy as np
from numpy import linalg as LA
import pandas as pd
import os

# - - - - - - - - - - - - - - - #
# Import the data
# - - - - - - - - - - - - - - - #

textFileNameInCurDir = sys.argv[1]
#textFileNameInCurDir = "magic04.txt"
fo = open("asign1.bmconrad.txt", "wb")

df = pd.read_csv(textFileNameInCurDir,
                 sep=",",
                 header=None,
                 names=["v1",
                        "v2",
                        "v3",
                        "v4",
                        "v5",
                        "v6",
                        "v7",
                        "v8",
                        "v9",
                        "v10",
                        "v11"],
                 dtype={'v1':  np.float64,
                        'v2':  np.float64,
                        'v3':  np.float64,
                        'v4':  np.float64,
                        'v5':  np.float64,
                        'v6':  np.float64,
                        'v7':  np.float64,
                        'v8':  np.float64,
                        'v9':  np.float64,
                        'v10': np.float64})
df = df.drop('v11',1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 1: 
# Subroutine to compute a matrix's covariance (Eq. 2.31)
# Confirmation that the subroutine acts as NumPy.cov()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def getCovariance(df):
    
    # - - - - - - - - - - - - - - #
    # Z = D - transpose(mu)*1     #
    # - - - - - - - - - - - - - - #
    
    # Get transpose(mu)*1; the mean matrix
    mu_T1 = df.mean()
    mu_T2 = pd.DataFrame([mu_T1],dtype=np.float64)
    mu_T_times_1 = pd.concat([mu_T2] * len(df.index))
    
    # Get D - tranpose(mu)*1; the centered matrix
    Z = np.subtract(df.as_matrix(),mu_T_times_1.as_matrix())    

    # Get tranpose(Z)*Z/n; the covariance matrix
    n_rows, n_cols = np.shape(Z)
    ZT = Z.transpose()
    ZTZ = np.dot(ZT,Z)
    cov_mat = np.true_divide(ZTZ,n_rows)
    
    # Return the covariance matrix object created; np.ndarray object
    return cov_mat

# Get the covariance matrix; from my function
my_cov = getCovariance(df)

# Get the covariance matrix; from NumPy
real_cov = np.cov(df.as_matrix().transpose(), bias=True)

# Compare the results; true
print("\n")
print("#1 --- Matching Covariance Matrices:")
print(np.allclose(my_cov,real_cov))
print("\n")
fo.write("\n")
fo.write("#1 --- Matching Covariance Matrices:")
fo.write(str(np.allclose(my_cov,real_cov)))
fo.write("\n")
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 2. 
# Get the two dominant eigenvectors
# Project the data onto this new eigenspace
# Print the variance of the new projected datapoints
# - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Get eigen vectors and values; default sorted
LAMBDA,U = LA.eig(my_cov)

# Sort eigen values, just in case
idx = LAMBDA.argsort()[::-1]   
LAMBDA = LAMBDA[idx]
U = U[:,idx]

# Create the new projection space
df_eig = pd.DataFrame({'eig0':U[0],
                        'eig1':U[1]})

# Project the old data onto the new eigen space
projectedData = pd.DataFrame(np.dot(df.as_matrix(),df_eig.as_matrix()))

# Print only the variance of this new vectorspace
projectedVariance = getCovariance(projectedData)
print("#2 -- Projected Variance Matrix on [eig0 eig1] Space:")
print(np.shape(projectedVariance))
print(projectedVariance)
print("\n")
fo.write("#2 -- Projected Variance Matrix on [eig0 eig1] Space:")
fo.write(str(np.shape(projectedVariance)))
np.savetxt('p2.txt', projectedVariance)
fo.write("\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 3.
# Print the covariance matrix Σ in its eigen-
#  decomposition form (UΛUT)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

#Use linalg.eig to find all the eigenvectors, and print the 

# Get eigen vectors and values; default sorted
my_cov = getCovariance(df)

# Get the eigen values and vectors
LAMBDA,U = LA.eig(my_cov)

# Format the decomposition and compute 
UT = U.transpose()
U_LAMBDA = U * LAMBDA
SIGMA = U_LAMBDA * UT

# Print the SIGMA
print("#3. Sigma in Decomposition Form:")
print(np.shape(SIGMA))
print(SIGMA)
print("\n")
fo.write("#3. Sigma in Decomposition Form:")
fo.write(str(np.shape(SIGMA)))
np.savetxt('p3.txt', SIGMA)
fo.write("\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 4:
# Write a subroutine to implement PCA Algorithm (Algorithm 7.1, Page 198).
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

def get_reduced_dimension(LAMBDA, alpha):
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Choose the smallest r s.t f(r) >= alpha given that
    #  f(r) = sum(i->r lambda_i)/sum(i->d lambda_i)
    #   when r = {1,2,...,d}
    #
    #
    #
    # So basically, pick the first appearance of r >= alpha.
    #
    #
    #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    
    # The denominator doesn't change.
    denominator = np.sum(LAMBDA)
    
    # Update a list of each quotient
    each_possible_r = []
    for r in range(len(LAMBDA)):
        numerator_i = np.sum([LAMBDA[i] for i in range(r)])
        each_possible_r.append(np.divide(numerator_i,denominator))
        
    # Grab the smallest r s.t r captures alpha*100 % variance
    smallest_r=0
    for i in range(len(each_possible_r)):
        if(each_possible_r[i] >= alpha):
            print "If r=",str(i),", then we capture ", str(alpha*100),"% variance.\n"
            smallest_r=i
            break
            
    # Return the smallest r with respect to alpha
    return smallest_r

def PCA(df, alpha):
    # - - - - - - - - - - - - - - - #
    coV = getCovariance(df)
    LAMBDA, U = LA.eig(coV)

    # Sort eigen values, just in case
    idx = LAMBDA.argsort()[::-1]   
    LAMBDA = U[idx]
    U = U[:,idx]

    # Find the reduced vectorspace of R^r
    r = get_reduced_dimension(LAMBDA, alpha)

    # compute the projection of the points from df onto 

    df_eig = pd.DataFrame({'eig'.join(str(dim)):U[dim]
                              for dim in range(0,r)})
    
    A = np.dot(df.as_matrix(),df_eig.as_matrix())
    
    # Return the projected data matrix onto the new vector space
    return(A)
    
print "#4-5 -- PCA Implementation and First 10 Observations Projection onto the Reduced Dimensionality Matrix with 90% Variance:"
fo.write("#4-5 -- PCA Implementation and First 10 Observations Projection onto the Reduced Dimensionality Matrix with 90% Variance:")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
# 5.
# Print the first 10 data points in the new set of basis vectors
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

df_onto_u_at_alpha_thresh = PCA(df, alpha=0.90)

# Print the first 10 data observations in our new vector space
# capturing 90% variance

print(df_onto_u_at_alpha_thresh[:10])
print("\n")
np.savetxt('p4-5.txt', df_onto_u_at_alpha_thresh[:10])
fo.write("\n")
fo.close()