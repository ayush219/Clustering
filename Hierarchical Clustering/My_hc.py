# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:22:11 2019

@author: Ayush
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset= pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Cluster
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('customer')
plt.ylabel('Euclidean distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y_hc= hc.fit_predict(X)

#Visualise
plt.scatter(X[Y_hc==0,0], X[Y_hc==0,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[Y_hc==1,0], X[Y_hc==1,1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[Y_hc==2,0], X[Y_hc==2,1], s=100, c='green', label='Cluster 2')
plt.scatter(X[Y_hc==3,0], X[Y_hc==3,1], s=100, c='cyan', label='Cluster 2')
plt.scatter(X[Y_hc==4,0], X[Y_hc==4,1], s=100, c='magenta', label='Cluster 2')
plt.title('Cluster of clients')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.show()