# Author: Blake Conrad
# Purpose: HW3 CSCI 48100
# File: Q1_bayes_train_write_bmconrad.py iris-shuffled.txt
# Constraints: Accepts training data as command line argument

from __future__ import division

# Import Libraries
import sys
import os
import numpy as np
import pandas as pd
import sys
import os
import math
from scipy import linalg

def multivariate_normal_pdf(x, mu, sigma):
    size = len(x)
    det = linalg.det(sigma)
    norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
    x_mu = np.matrix(x - mu)
    inv = sigma.I
    result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return norm_const * result

def normal_pdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(math.sqrt(2*math.pi)*abs(sigma)))*math.exp(-u*u/2)
    return y


class Bayes_Classifier:

        
    # Methods    
    @classmethod
    def build(self, df_inc):
        
        # Flag Appropriately
        self.built = True
        self.DF_TRAIN = df_inc
        self.classes = pd.unique(self.DF_TRAIN["Flower Type"])
        self.actual_labels = self.DF_TRAIN.ix[:,"Flower Type"]
        self.k = len(self.classes)
        self.n = len(self.DF_TRAIN)
        
        # Containers
        self.D_i = list()
        self.n_i = list()
        self.P_c_i = list()
        self.mean_i = list()
        self.sigma_i = list()
        
        # Algorthm 18.1
        for i in range(self.k):
            self.D_i.append(self.DF_TRAIN.loc[self.DF_TRAIN['Flower Type'] == self.classes[i]])
            self.n_i.append(len(self.D_i[i]))
            self.P_c_i.append(self.n_i[i] / self.n)
            self.mean_i.append(self.D_i[i].mean())
            self.sigma_i.append(np.cov(self.D_i[i].ix[:,:4].as_matrix().transpose(), bias=True))
            
            """
            print "D_i\n", D_i[i]
            print "n_i\n", n_i[i]
            print "P(c_i)\n", P_c_i[i]
            print "mean_i\n", mean_i[i]
            print "sigma_i\n", sigma_i[i]
            """
        # Return the model
        return self.D_i, self.n_i, self.P_c_i, self.mean_i, self.sigma_i
    
    @classmethod
    def writeModel(self):
        
        # Round to 2 decimals as requested
        self.P_c_i = np.around(self.P_c_i, decimals=2)
        self.mean_i = np.around(self.mean_i, decimals=2)
        self.sigma_i = np.around(self.sigma_i, decimals=2)
        
        target = open("bayes_model.txt", 'w')
        target.write("--- skip this line --- P(c_i) is each line per class~\n")
        target.write(str(self.P_c_i[0]))
        target.write("\n")
        target.write(str(self.P_c_i[1]))
        target.write("\n")
        target.write(str(self.P_c_i[2]))
        target.write("\n")
        target.write("--- skip this line --- classes[i] -- each class per line\n")
        target.write(str(self.classes[0]))
        target.write("\n")
        target.write(str(self.classes[1]))
        target.write("\n")
        target.write(str(self.classes[2]))
        target.write("\n")
        target.write("--- skip this line --- mean_i is each line per class~\n")
        for mu in self.mean_i:
            target.write(str(mu.tolist()[0]))
            target.write(",")
            target.write(str(mu.tolist()[1]))
            target.write(",")
            target.write(str(mu.tolist()[2]))
            target.write(",")
            target.write(str(mu.tolist()[3]))
            target.write("\n")
        target.write("--- skip this line --- sigma_i is each 4 lines per class~\n")
        for coV in self.sigma_i: #1,2,and3
            for v in coV:        #row1,2,3,and4
                for rowVal in v: #col1,2,3,and4
                    target.write(str(rowVal))
                    target.write(",")
                target.write("\n") 
        
        print "Model Object Written to: bayes_model.txt"
        target.close()



"""
Should be a 3x3 matrix of the following shape

                                    PREDICTED

                          setosa   versicolor   virginica
                          +-------+------------+---------+
               setosa     |       |            |         |
                          +-------+------------+---------+
    ACTUAL     versicolor |       |            |         |
                          +-------+------------+---------+
               versicolor |       |            |         | 
                          +-------+------------+---------+

Simple Example:
Assuming a sample of 27 animals 8 cats, 6 dogs, and 13 rabbits, the resulting confusion matrix could look like the table below:

If the sample had 27 classifiers given, 8 cats, 6 dogs, and 13 rabbits,
the corresponding confusion matrix would be the following:
                Cat	Dog	Rabbit
	    Cat	    5	3	0
        Dog	    2	3	1
        Rabbit	0	2	11

Assuming the above confusion matrix, the following table of confusion for the cat class would be the following:

5 true positives (actual cats that were correctly classified as cats)

3 false negatives (cats that were incorrectly marked as dogs)

2 false positives (dogs that were incorrectly labeled as cats)

17 true negatives (all the remaining animals, correctly classified as non-cats)

Cat
[[5  3],
 [2 17]]

Dog
[[3 1],
 [2 21]]
 
Rabbit
[[11 0],
 [0 16]]

So there is a confusion matrix per variable in the data set as to how well it predicted.
"""

def main():
    
    # Import Data
    trainName = "data/"
    trainName += sys.argv[1] #iris-shuffled.txt

    print "Training Data File Name:" , trainName
    
    # Format Data
    df_train = pd.read_csv(trainName,
                          sep=",",
                          names=["Septal Length",
                                "Septal Width",
                                "Pedal Length",
                                "Pedal Width",
                                "Flower Type"],
                          dtype={'Septal Length':  np.float64,
                                 'Septal Width' :  np.float64,
                                 'Pedal Length' :  np.float64,
                                 'Pedal Width'  :  np.float64})


    # BAYES CLASSIFIER
    D_i, n_i, P_c_i, mean_i, sigma_i = Bayes_Classifier.build(df_train)

    # Train this using 3-fold CV
    Bayes_Classifier.writeModel()


if __name__ == "__main__":
    main()
