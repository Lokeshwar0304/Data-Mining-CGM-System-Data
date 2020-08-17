# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:33:17 2020

@author: Lokeshwar
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from Adaboost import features
import pickle

file_name = input ('Enter the testing file name:\n' )


def fn_testing(name):
    
    with open("AdaBoostClassifier_Model.pkl", 'rb') as file:
        AdaBoostClassifier_Model = pickle.load(file) 
        test_df = pd.read_csv(name, header=None)
    
    with open("pca_meal.pkl", 'rb') as file:
        pca = pickle.load(file)
        
    abc=features(test_df)
    test_ss = StandardScaler().fit_transform(abc)
    test_pca=pca.fit_transform(test_ss)
        
    predictions = AdaBoostClassifier_Model.predict(test_pca)
    print(predictions)
    pd.DataFrame(predictions).to_csv("test_output_classes.csv")
    
    
fn_testing(file_name)
