# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 06:26:48 2020

@author: Lokeshwar
"""
# In this, the test dataset points are assigned to KMEANS clusters and DBSCAN clusters based on KNN Classifer.

import pandas as pd
from train import features
from sklearn.neighbors import KNeighborsClassifier

file_name = input ('Enter the testing file name:\n' )
#k,distance_metric = input ('Enter K value and distance metric (Eg: 5 Manhattan) -' ).split()



kmeans_labels=[]
dbscan_labels=[]
    
def KNN(k,test_features):
    train_features=pd.read_csv('train_features.csv') 
    knn_kmeans=KNeighborsClassifier(n_neighbors=k,p=2)
    knn_dbscan=KNeighborsClassifier(n_neighbors=k,p=2)
    knn_kmeans.fit(train_features.iloc[:,:-3],train_features.iloc[:,-2])
    knn_dbscan.fit(train_features.iloc[:,:-3],train_features.iloc[:,-1])
   
    
    for row in range(0,test_features.shape[0]):
        kmeans_labels.append(knn_kmeans.predict([list(test_features.iloc[row,:])])[0])
        dbscan_labels.append(knn_dbscan.predict([list(test_features.iloc[row,:])])[0])
        

def fn_testing(file_name):
    test_df = pd.read_csv(file_name,names=[i for i in range(30)],index_col=False)        
    test_features=features(test_df)     
    k=25
    KNN(k,test_features)
    
    results_df=pd.DataFrame({'DBSCAN':dbscan_labels,'KMEANS':kmeans_labels})
    results_df.to_csv('P3labels.csv',header=False)
    print(results_df)
    
    
    
  
fn_testing(file_name)
