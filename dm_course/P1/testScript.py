
# coding: utf-8

import sys
import os
import scipy
import numpy as np
import pandas as pd

df_train = pd.read_csv("/Users/bmc/Desktop/CSCI-48100/P1/data/iris-shuffled.txt",
                      sep=",",
                      names=["Septal Length",
                            "Septal Width",
                            "Pedal Length",
                            "Pedal Width",
                            "Flower Type"])
df_test = pd.read_csv("/Users/bmc/Desktop/CSCI-48100/P1/data/iris.txt",
                      sep=",",
                      names=["Septal Length",
                            "Septal Width",
                            "Pedal Length",
                            "Pedal Width"])

df_test["Flower Type"] = " "
print df_train.head()
print df_test.head()

df_combined = pd.concat([df_train, df_test])

df_combined.head()

df_combined["id"] = [i for i in range(300)]
df_combined.head()
cols = df_combined.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_combined_reordered = df_combined[cols]




