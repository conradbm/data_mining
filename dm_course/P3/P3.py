
# coding: utf-8

# <h2>Bayesian Classifiers</h2>
# <ul>
# <li><h4>Author: Blake Conrad</h4></li>
# <li><h4>Purpose: P3 for CSCI 48100 </h4></li>
# <li><h4>Goal: Implement Bayes and Naive Bayes Classifiers </h4></li>
# </ul>

# In[1]:

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
    
    # Class Attributes
    constructed = False
    built = False
    predicted = False
    
    # Constructor
    def __init__(self, DF_TRAIN):
        
        # Flag Appropriately
        self.constructed = True
        
        # Save training set
        self.DF_TRAIN = DF_TRAIN
        
        # Class labels
        self.classes = pd.unique(self.DF_TRAIN["Flower Type"])
        self.actual_labels = self.DF_TRAIN.ix[:,"Flower Type"]
        # Constants
        self.k = len(self.classes)
        self.n = len(self.DF_TRAIN)

        # Containers
        self.D_i = list()
        self.n_i = list()
        self.P_c_i = list()
        self.mean_i = list()
        self.sigma_i = list()
        
        
    # Methods    
    def build(self):
        
        # Flag Appropriately
        self.built = True
        
        # Algorthm 18.1
        for i in range(self.k):
            self.D_i.append(self.DF_TRAIN.loc[self.DF_TRAIN['Flower Type'] == self.classes[i]])
            self.n_i.append(len(self.D_i[i]))
            self.P_c_i.append(self.n_i[i] / self.n)
            self.mean_i.append(self.D_i[i].mean())
            self.sigma_i.append(np.cov(self.D_i[i].ix[:,:4].as_matrix().transpose(), bias=True))
            """
            print "D_i\n", D[i]
            print "n_i\n", n_i[i]
            print "P(c_i)\n", P_c_i[i]
            print "mean_i\n", mean_i[i]
            print "sigma_i\n", sigma_i[i]
            """
            
        # Return the model
        return self.D_i, self.n_i, self.P_c_i, self.mean_i, self.sigma_i
    
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

        target.close()
        
    def readModel(self):
        
        
        target = open("bayes_model.txt","r")
        lines = target.readlines()
        
        # P(c_i)
        label1 = lines[0]
        P_c_i_str = lines[1:4]
        P_c_i_flt = map(float, P_c_i_str)
        P_c_i_ls = P_c_i_flt
        P_c_i_ls = np.array(P_c_i_ls)
        
        # Mean_i
        label2 = lines[4]
        mu1_str = lines[5].split(",")
        mu2_str = lines[6].split(",")
        mu3_str = lines[7].split(",")
        mu1_flt = map(float, mu1_str)
        mu2_flt = map(float, mu2_str)
        mu3_flt = map(float, mu3_str)
        mean_i_ls = [mu1_flt, mu2_flt, mu3_flt]
        mean_i_ls = map(np.array, mean_i_ls)
        
        label3 = lines[8]
        cov1_str = lines[9:13] #4 lines
        cov2_str = lines[13:17] #4 lines
        cov3_str = lines[17:] #4 lines

        
        # Cov1
        s1 = cov1_str[0][:-2].split(",")
        s2 = cov1_str[1][:-2].split(",")
        s3 = cov1_str[2][:-2].split(",")
        s4 = cov1_str[3][:-2].split(",")
        cov1_str = [s1, s2, s3, s4]
        cov1_mat = np.matrix(cov1_str, dtype=np.float)
        
        # Cov2
        s1 = cov2_str[0][:-2].split(",")
        s2 = cov2_str[1][:-2].split(",")
        s3 = cov2_str[2][:-2].split(",")
        s4 = cov2_str[3][:-2].split(",")
        cov2_str = [s1, s2, s3, s4]
        cov2_mat = np.matrix(cov2_str, dtype=np.float)
        
        # Cov3
        s1 = cov3_str[0][:-2].split(",")
        s2 = cov3_str[1][:-2].split(",")
        s3 = cov3_str[2][:-2].split(",")
        s4 = cov3_str[3][:-2].split(",")
        cov3_str = [s1, s2, s3, s4]
        cov3_mat = np.matrix(cov3_str, dtype=np.float)
        
        sigma_i_ls = [cov1_mat, cov2_mat, cov3_mat]
        
        self.P_c_i = P_c_i_ls
        self.mean_i = mean_i_ls
        self.sigma_i = sigma_i_ls
        return self.P_c_i, self.mean_i , self.sigma_i
    
    def predict(self, DF_TEST):
         
        # Flag Appropriately
        self.predicted = True
        
        # Save the testing set
        self.DF_TEST = DF_TEST
        
        # Containers
        self.predicted_labels = list()
        
        # For each point in DF_TEST
        for j in range(len(self.DF_TEST)):
            
            # Get the maxmimum probability classification
            max_probability_class_label = ""
            max_probability = 0
            for i in range(self.k):
                tmp = multivariate_normal_pdf(self.DF_TEST.ix[j,:4].as_matrix(),
                                              self.mean_i[i],
                                              self.sigma_i[i])
                tmp = tmp * self.P_c_i[i]
                
                if(tmp > max_probability):
                    max_probability = tmp
                    max_probability_class_label = self.classes[i]
                    
            # Store our prediction for each point
            self.predicted_labels.append(max_probability_class_label)

        self.actual_labels = self.DF_TEST.ix[:,"Flower Type"].tolist()
        # Return the predictions
        return self.predicted_labels

    def get_confusion(self):
        if([self.constructed, self.built, self.predicted]):
            print "Safe to calculate."
            
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
            self.classes = ["Iris-virginica","Iris-versicolor","Iris-setosa"]
            for j in range(len(self.predicted_labels)):
                if self.predicted_labels[j] == self.actual_labels[j]:#CORRET
                    if(self.predicted_labels[j] == "Iris-virginica"):
                        virginica_virginica+=1
                    elif(self.predicted_labels[j] == "Iris-versicolor"):
                        versicolor_versicolor+=1
                    elif(self.predicted_labels[j] == "Iris-setosa"):
                        setosa_setosa+=1
                else: #incorrect
                    if (self.predicted_labels[j] == "Iris-virginica" and self.actual_labels[j] == "Iris-versicolor"):
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
                    

            confusion_matrix = np.matrix([[virginica_virginica,virginica_versicolor,virginica_setosa],
                                          [versicolor_virginica,versicolor_versicolor,versicolor_setosa],
                                          [setosa_virginica,setosa_versicolor,setosa_setosa]],
                                         dtype=np.int32).T
            
            # Record TP, FP, FN, TN for each class
            # If we are looking at our 3x3 confusion matrix, we can construct it using the following logic
            # [[currentMainDiagonalEntry, sum of rows beneath mainDiagonalEntry],
            #  [sum of cols beside currentMainDiagonalEntry, rest of mainDiagonalEntries]]
            # Predicted correctly
            
            print "Confusion Matrix Matrix"
            print confusion_matrix

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

            print "Virginica Confusion Matrix"
            print virginica_confusion
            print

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
                
            print "Versicolor Confusion Matrix"
            print versicolor_confusion
            print
            
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
            print "Setosa Confusion Matrix"
            print setosa_confusion
            print
            #print self.predicted_labels
            #print self.actual_labels
            
            return(virginica_confusion, versicolor_confusion, setosa_confusion)
            
        else:
            print "Not safe to calculate. Consider building and predicting with your model first."


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
Assuming a sample of 27 animals — 8 cats, 6 dogs, and 13 rabbits, the resulting confusion matrix could look like the table below:

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

class Naive_Bayes_Classifier:
    
    # Class Attributes
    constructed = False
    built = False
    predicted = False
    
    # Constructor
    def __init__(self, DF_TRAIN):
        
        # Flag Appropriately
        self.constructed = True
        
        # Save training set
        self.DF_TRAIN = DF_TRAIN
        
        # Class labels
        self.classes = pd.unique(self.DF_TRAIN["Flower Type"])
        self.actual_labels = self.DF_TRAIN.ix[:,"Flower Type"]
        
        # Constants
        self.k = len(self.classes)
        self.n = len(self.DF_TRAIN)

        # Containers
        self.D_i = list()
        self.n_i = list()
        self.P_c_i = list()
        self.mean_i = list()
        self.sigma_i = list()
        
        
    # Methods    
    def build(self):
        
        # Flag Appropriately
        self.built = True
        
        # Algorthm 18.2
        for i in range(self.k):
            self.D_i.append(self.DF_TRAIN.loc[self.DF_TRAIN['Flower Type'] == self.classes[i]])
            self.n_i.append(len(self.D_i[i]))
            self.P_c_i.append(self.n_i[i] / self.n)
            self.mean_i.append(self.D_i[i].mean())
            
            variance_i = list()
            for j in range(len(self.DF_TRAIN.columns) - 1):
                variance_i.append(np.var(self.D_i[i].ix[:,j].as_matrix(), out=None))
                #print variance_i[j]
                #raw_input(",,,")
            self.sigma_i.append(variance_i)
            
            """
            print "D_i\n", self.D[i]
            print "n_i\n", self.n_i[i]
            print "P(c_i)\n", self.P_c_i[i]
            print "mean_i\n", self.mean_i[i][:self.DF_TRAIN.columns]
            print "sigma_i\n", self.sigma_i[i][:self.DF_TRAIN.columns]
            """
            
        # Return the model
        return self.D_i, self.n_i, self.P_c_i, self.mean_i, self.sigma_i

    
    def writeModel(self):
        
        # Round to 2 decimals as requested
        self.P_c_i = np.around(self.P_c_i, decimals=2)
        self.mean_i = np.around(self.mean_i, decimals=2)
        self.sigma_i = np.around(self.sigma_i, decimals=2)
        
        target = open("naive_bayes_model.txt", 'w')
        target.write("--- skip this line --- P(c_i) is each line per class~\n")
        target.write(str(self.P_c_i[0]))
        target.write("\n")
        target.write(str(self.P_c_i[1]))
        target.write("\n")
        target.write(str(self.P_c_i[2]))
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
        target.write("--- skip this line --- sigma_i is each lines per class~\n")
        for coV in self.sigma_i: #1,2,and3
            target.write(str(coV.tolist()[0]))
            target.write(",")
            target.write(str(coV.tolist()[1]))
            target.write(",")
            target.write(str(coV.tolist()[2]))
            target.write(",")
            target.write(str(coV.tolist()[3]))
            target.write("\n")
            
        target.close()
        
    def readModel(self):
        
        
        target = open("naive_bayes_model.txt","r")
        lines = target.readlines()
        
        # P(c_i)
        label1 = lines[0]
        P_c_i_str = lines[1:4]
        P_c_i_flt = map(float, P_c_i_str)
        P_c_i_ls = P_c_i_flt
        P_c_i_ls = np.array(P_c_i_ls)
        
        # Mean_i
        label2 = lines[4]
        mu1_str = lines[5].split(",")
        mu2_str = lines[6].split(",")
        mu3_str = lines[7].split(",")
        mu1_flt = map(float, mu1_str)
        mu2_flt = map(float, mu2_str)
        mu3_flt = map(float, mu3_str)
        mean_i_ls = [mu1_flt, mu2_flt, mu3_flt]
        mean_i_ls = map(np.array, mean_i_ls)
        
        label3 = lines[8]
        cov1_str = lines[9]
        cov2_str = lines[10]
        cov3_str = lines[11]
        
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
        self._mean_i = mean_i_ls
        self.sigma_i = sigma_i_ls
        
        return self.P_c_i, self._mean_i, self.sigma_i
        
    def predict(self, DF_TEST):
         
        # Flag Appropriately
        self.predicted = True
        
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
    
    def get_confusion(self):
        if([self.constructed, self.built, self.predicted]):
            print "Safe to calculate."
            
        else:
            print "Not safe to calculate. Consider building and predicting with your model first."



# <h2> Problem 1 <h3>Build the model file</h3></h2>
# <ul> 
# <li>Implement a python program that accepts a dataset as a command line parameter and generates
# a model file in the current directory. </li>
# <li>The model file contains: (i) the prior probabilities of each of
# the classes;</li>
# <li>(ii) the mean and the covariance matrix of each of the classes. Our objective is to use
# this model file to perform classification using full Bayes classification method.</li>
# <li>To ensure readabilityof the model file, please write all the numeric values using 2 digits after the decimal point. Youcan use build-in functions in the NumPy package for computing the mean and the covariance.</li>
# </ul>
# 

# <h2>Problem 2</h2>
# <h3>Testing the model</h3>
# <ul>
# <li>Implement a python program that accepts a model file (output of Q1) and a test file as command
# line parameter.</li>
# <li>The test file has identical format of the train file.</li>
# <li>For each instance of the test file,the program outputs the predicted label.</li>
# <li>The program also prints a confusion matrix by comparingthe true labels and predicted labels of all the instances.</li>
# </ul>
#   

# <h2>Problem 3</h2>
# <h3>3-fold Cross Validation</h3>
# <ul>
# <li>For this, make 3-folds of the file iris.txt.shuffled by considering 50 consecutive instances as one fold
# 1(do not reorder the instances in the files).</li>
# <li>Use the program from Q1 for training purpose using
# instances from two of the folds and use the program from Q2 for testing on the instances of the
# remaining fold.</li>
# <li>Print the confusion matrix for each of the three folds (when they were used as
# test). Also, for each class, print the accuracy, precision, recall, and F-score, averaged over 3-folds.</li>
# </ul>
# 

# <h2> Problem 4</h2>
# <h3> Repeat 1-3 for Naive Bayes</h3>

# In[ ]:




# In[ ]:

def main():
    
    # Import Data
    trainName = "data/" + sys.argv[1] #iris-shuffled.txt
    testName = "data/" + sys.argv[2] #iris.txt

    print "Training Data:" , trainName
    print "Testing Data:", testName
    
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



    # Cleaning Data
    print df_train.head()
    print df_test.head()

    # BAYES CLASSIFIER
    bayes_classifier = Bayes_Classifier(df_train)
    D_i, n_i, P_c_i, mean_i, sigma_i = bayes_classifier.build()

    #print "D_i:\n", D_i
    #print "n_i:\n", n_i
    #print "P_c_i:\n", P_c_i
    #print "mean_i:\n", mean_i
    #print "sigma_i:\n", sigma_i
    bayes_classifier.writeModel()
    bayes_classifier.readModel()

    # Predict with the built model object
    prediction_labels = bayes_classifier.predict(df_test)
    bayes_classifier.get_confusion()
    #print prediction_labels
    # Get how well the model did
    # accuracy = bayes_classifier.get_confusion()


    """
    # NAIVE BAYES CLASSIFIER
    naive_bayes_classifier = Naive_Bayes_Classifier(df_train)
    D_i, n_i, P_c_i, mean_i, sigma_i = naive_bayes_classifier.build()

    #naive_bayes_classifier.writeModel()
    #naive_bayes_classifier.readModel()

    # Predict on the unseen data
    #predicted_labels = naive_bayes_classifier.predict(df_test)
    print
    print predicted_labels

    # Determine the accuracy of the model
    #accuracy = naive_bayes_classifier.get_accuracy()
"""

if __name__ == "__main__":
    print "Starting Program."
    main()
    print "Finished Program."
