# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:14:14 2019

@author: Ayush
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset= pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,3:5].values

#Cluster
from sklearn.cluster import KMeans
WCSS=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,11), WCSS)
plt.title('Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
Y_kmeans=kmeans.fit_predict(X)

#Visualise
plt.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],s=100,c='red',label='cluster 1')
plt.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],s=100,c='green',label='cluster 3')
plt.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],s=100,c='magenta',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Cluster of clients')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.show()