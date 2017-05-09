# Author: Blake Conrad
# Purpose: HW3 CSCI 48100
# File: Q2_naive_bayes_read_test_bmconrad.py iris-shuffled.txt
# Constraints: Accepts testing data as command line argument

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


class Naive_Bayes_Classifier:


    @classmethod
    def readModel(self):
        
        target = open("naive_bayes_model.txt","r")
        lines = target.readlines()
        
        # P(c_i)
        label1 = lines[0]
        P_c_i_str = lines[1:4]
        P_c_i_flt = map(float, P_c_i_str)
        P_c_i_ls = P_c_i_flt
        P_c_i_ls = np.array(P_c_i_ls)
        
        class0 = lines[5][:-1]
        class1 = lines[6][:-1]
        class2 = lines[7][:-1]
        self.classes = [class0, class1, class2] 
        
        # Mean_i
        label2 = lines[8]

        mu1_str = lines[9].split(",")
        mu2_str = lines[10].split(",")
        mu3_str = lines[11].split(",")
        mu1_flt = map(float, mu1_str)
        mu2_flt = map(float, mu2_str)
        mu3_flt = map(float, mu3_str)
        mean_i_ls = [mu1_flt, mu2_flt, mu3_flt]
        mean_i_ls = map(np.array, mean_i_ls)
        
        label3 = lines[12]
        cov1_str = lines[13]
        cov2_str = lines[14]
        cov3_str = lines[15]
        
        # Var1
        cov1_str = cov1_str[:-1].split(",")
        cov1_arr = np.array(cov1_str, dtype=np.float)
        
        # Var1
        cov2_str = cov2_str[:-1].split(",")
        cov2_arr = np.array(cov2_str, dtype=np.float)
        
        # Var3
        cov3_str = cov3_str[:-1].split(",")
        cov3_arr = np.array(cov3_str, dtype=np.float)
        
        sigma_i_ls = [cov1_arr, cov2_arr, cov3_arr]
        
        self.P_c_i = P_c_i_ls
        self.mean_i = mean_i_ls
        self.sigma_i = sigma_i_ls

        return self.P_c_i, self.mean_i, self.sigma_i
        
    @classmethod
    def predict(self, DF_TEST):

        # Constants
        self.k = len(self.classes)
        
        # Save the testing set
        self.DF_TEST = DF_TEST
        
        # Containers
        self.predicted_labels = list()
        
        # For each point in DF_TEST
        for j in range(len(DF_TEST)):
            
            # Get the maxmimum probability classification
            max_probability_class_label = ""
            max_probability = 0
            for i in range(self.k):
                
                product_of_columns_probability = 1
                
                for r in range(len(self.DF_TEST.columns)-1):
                    product_of_columns_probability *= normal_pdf(self.DF_TEST.ix[j,r],
                                                                             self.mean_i[i][r],
                                                                             self.sigma_i[i][r])
                
                tmp = product_of_columns_probability * self.P_c_i[i]
                if(tmp > max_probability):
                    max_probability = tmp
                    max_probability_class_label = self.classes[i]
                    
            # Store our prediction for each point
            self.predicted_labels.append(max_probability_class_label)
            
        self.actual_labels = self.DF_TEST.ix[:,4]
        # Return the predictions
        return self.predicted_labels
    

    @classmethod
    def get_confusion(self):
        
        virginica_virginica=0
        virginica_versicolor=0
        virginica_setosa=0

        versicolor_virginica=0
        versicolor_versicolor=0
        versicolor_setosa=0

        setosa_virginica=0
        setosa_versicolor=0
        setosa_setosa=0

        # c1 'Iris-virginica'
        # c2 'Iris-versicolor'
        # c3 'Iris-setosa'

        for j in range(len(self.predicted_labels)):
            if(self.predicted_labels[j] == "Iris-virginica" and self.actual_labels[j] == "Iris-virginica"):
                virginica_virginica+=1
            elif(self.predicted_labels[j] == "Iris-versicolor" and self.actual_labels[j] == "Iris-versicolor"):
                versicolor_versicolor+=1
            elif(self.predicted_labels[j] == "Iris-setosa" and self.actual_labels[j] == "Iris-setosa"):
                setosa_setosa+=1
            elif (self.predicted_labels[j] == "Iris-virginica" and self.actual_labels[j] == "Iris-versicolor"):
                virginica_versicolor+=1
            elif(self.predicted_labels[j] == "Iris-virginica" and self.actual_labels[j] == "Iris-setosa"):
                virginica_setosa+=1
            elif(self.predicted_labels[j] == "Iris-versicolor" and self.actual_labels[j] == "Iris-virginica"):
                versicolor_virginica+=1
            elif(self.predicted_labels[j] == "Iris-versicolor" and self.actual_labels[j] == "Iris-setosa"):
                versicolor_setosa+=1
            elif(self.predicted_labels[j] == "Iris-setosa" and self.actual_labels[j] == "Iris-virginica"):
                setosa_virginica+=1
            elif(self.predicted_labels[j] == "Iris-setosa" and self.actual_labels[j] == "Iris-versicolor"):
                setosa_versicolor+=1
            else: print "Something went wrong with label: ", self.predicted_labels[j]



        confusion_matrix = np.matrix([[virginica_virginica, virginica_versicolor, virginica_setosa],
                                      [versicolor_virginica, versicolor_versicolor, versicolor_setosa],
                                      [setosa_virginica, setosa_versicolor, setosa_setosa]],
                                     dtype=np.int32)

        # Record TP, FP, FN, TN for each class
        # If we are looking at our 3x3 confusion matrix, we can construct it using the following logic
        # [[currentMainDiagonalEntry, sum of rows beneath mainDiagonalEntry],
        #  [sum of cols beside currentMainDiagonalEntry, rest of mainDiagonalEntries]]
        # Predicted correctly

        print "Confusion Matrix Matrix Of All Variables:"
        print confusion_matrix, "\n"

        row1 = np.array(confusion_matrix[0])[0]
        row2 = np.array(confusion_matrix[1][0])[0]
        row3 = np.array(confusion_matrix[2][0])[0]
        #print row1
        #print row2
        #print row3

        TP_virginica = row1[0]
        FP_virginica = sum([row2[0], row3[0]])
        FN_virginica = sum([row1[1] , row1[2]])
        TN_virginica = sum([row2[1], row3[2]])



        #print "TP, FP, FN, TN"
        #print TP_virginica
        #print FP_virginica
        #print FN_virginica
        #print TN_virginica

        virginica_confusion = np.matrix([[TP_virginica, FP_virginica],
                                         [FN_virginica, TN_virginica]], dtype=np.int32)

        print "Virginica Confusion Matrix:"
        print virginica_confusion, "\n"

        acc = (TP_virginica + TN_virginica) / (TP_virginica + FP_virginica + FN_virginica + TN_virginica)
        prec = (TP_virginica) / (TP_virginica +  FP_virginica)
        recc = (TP_virginica) / (TP_virginica + FN_virginica)
        fscore = (2*prec*recc) / (prec + recc) #2PR / (P + R)

        print "Accuracy: ", str(acc), "\t", 
        print "Precision: ", str(prec), "\t",
        print "Recall: ", str(recc), "\t",
        print "F-score: ", str(fscore), "\t\n",

        TP_versicolor = row2[1]
        FP_versicolor = sum([row1[1], row3[1]])
        FN_versicolor = sum([row2[0] , row2[2]])
        TN_versicolor = sum([row1[0], row3[2]])



        #print "TP, FP, FN, TN"
        #print TP_versicolor
        #print FP_versicolor
        #print FN_versicolor
        #print TN_versicolor

        versicolor_confusion = np.matrix([[TP_versicolor, FP_versicolor],
                                             [FN_versicolor, TN_versicolor]], dtype=np.int32)

        print "Versicolor Confusion Matrix:"
        print versicolor_confusion, "\n"


        acc = (TP_versicolor + TN_versicolor) / (TP_versicolor + FP_versicolor + FN_versicolor + TN_versicolor)
        prec = (TP_versicolor ) / (TP_versicolor  +  FP_versicolor )
        recc = (TP_versicolor ) / (TP_versicolor  + FN_versicolor )
        fscore = (2*prec*recc) / (prec + recc) #2PR / (P + R)

        print "Accuracy: ", str(acc) , "\t",
        print "Precision: ", str(prec), "\t",
        print "Recall: ", str(recc), "\t",
        print "F-score: ", str(fscore), "\t\n",

        TP_setosa = row3[2]
        FP_setosa = sum([row2[2], row1[2]])
        FN_setosa = sum([row3[0], row3[1]])
        TN_setosa = sum([row1[0], row2[1]])


        #print "TP, FP, FN, TN"
        #print TP_setosa
        #print TP_setosa
        #print FN_setosa
        #print TN_setosa

        setosa_confusion = np.matrix([[TP_setosa, FP_setosa],
                                          [FN_setosa, TN_setosa]], dtype=np.int32)
        print "Setosa Confusion Matrix:"
        print setosa_confusion, "\n"

        acc = (TP_setosa + TN_setosa) / (TP_setosa + FP_setosa + FN_setosa + TN_setosa)
        prec = (TP_setosa ) / (TP_setosa  +  FP_setosa )
        recc = (TP_setosa ) / (TP_setosa  + FN_setosa )
        fscore = (2*prec*recc) / (prec + recc) #2PR / (P + R)

        print "Accuracy: ", str(acc), "\t",
        print "Precision: ", str(prec), "\t",
        print "Recall: ", str(recc), "\t",
        print "F-score: ", str(fscore), "\t\n",
        #print self.predicted_labels
        #print self.actual_labels

        return(virginica_confusion, versicolor_confusion, setosa_confusion)



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
    testName = "data/"
    testName += sys.argv[1] #iris-shuffled.txt

    # Format Data
    df_test = pd.read_csv(testName,
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


    Naive_Bayes_Classifier.readModel()
    
    prediction_labels = Naive_Bayes_Classifier.predict(df_test)
    print "Predicted Labels: ", prediction_labels    
    Naive_Bayes_Classifier.get_confusion()


if __name__ == "__main__":
    main()
