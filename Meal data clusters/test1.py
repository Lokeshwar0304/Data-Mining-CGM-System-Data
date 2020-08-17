# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 06:26:48 2020

@author: Lokeshwar
"""
# In this, the dataset points are assigned to its nearest KMEANS and DBSCAN clusters using euclidean distance 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from train import features
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.metrics import classification_report
from statistics import mode



#print(classification_report(df['bins'], df['pLabels']))
op_kmeans_labels=[]
op_dbscan_labels=[]
    

    

def fn_testing():
    test_df = pd.read_csv('proj3_test.csv')        
    test_features=features(test_df)
    train_features=pd.read_csv('train_features.csv') 
    matrix=pd.DataFrame() 
    for i in range(0,test_features.shape[0]):
        sample_distance=[]
        test_row=test_features.loc[i]
        for j in range(0,train_features.shape[0]):
            train_row=train_features.loc[j][:-2]
            sample_distance.append(distance.euclidean(list(test_row),list(train_row)))
        matrix[str(i)]=sample_distance
    
    matrix['kmeans_labels']=train_features['kmeans_cluster']
    matrix['dbscan_labels']=train_features['dbscan_cluster']
    
    k=25
    sorted_list=[]
    kmeans=[]
    dbscan=[]
    for i in range(0,matrix.shape[1]-2):
        sorted_list=list(matrix[str(i)])
        sorted_list.sort()
        for j in range (0,k):
            a=matrix[matrix[str(i)]==sorted_list[j]]
            kmeans.append(int(a['kmeans_labels'].to_string()[len(a['kmeans_labels'].to_string())-1]))
            dbscan.append(int(a['dbscan_labels'].to_string()[len(a['dbscan_labels'].to_string())-1]))
        op_kmeans_labels.append(mode(kmeans))
        op_dbscan_labels.append(mode(dbscan))
        
    output_df=pd.DataFrame({'dbscan':op_dbscan_labels,'kmeans':op_kmeans_labels})  
    print(output_df)
  
fn_testing()
