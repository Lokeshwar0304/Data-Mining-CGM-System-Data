
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:52:56 2020

@author: Lokeshwar
"""
import numpy as np
from scipy.stats import skew
import pandas as pd
from scipy.fftpack import fft
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score,accuracy_score
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


fileMatch = ['mealData*.csv', 'Nomeal*.csv'] # Expected format of the mealData and Nomeal data CSV files
meal_df = pd.DataFrame()
noMeal_df = pd.DataFrame()


#################################### PRE PROCESSING #############################################################
#pre processing to remove NULLs within in a certain threshold and fill the remaining using interpolation.

def fn_preprocessing(data):
    temp_df = data
    temp_df.index = [i for i in range(temp_df.shape[0])]
    mrow_index = []
    for row in range(0, temp_df.shape[0]):
        if temp_df.iloc[row, :].isnull().sum() > 15: # remove the row if the number of NULLs are >15
            mrow_index.append(row)

    mcolumn_index = []
    for column in range(0, temp_df.shape[1]):
        if temp_df.iloc[:, column].isnull().sum() > 30: # remove column if the number of NULLs are >30
            mcolumn_index.append(column)

    df = temp_df.drop(index=mrow_index, columns=[temp_df.columns[c] for c in mcolumn_index])
    df.index = [i for i in range(df.shape[0])]
    
    df=df.interpolate(axis=0,method='quadratic') # Replacing remaining NaN values

    ppFile = 'preprocessed_' + str(fileMatch.index(name)) + '.csv'
    df.to_csv(ppFile, index=False)
    
for name in fileMatch:
    df_csv = pd.concat([pd.read_csv(file, names=[i for i in range(31)]) for file in glob.glob(name)])
    fn_preprocessing(df_csv)

#combined meal data and the no-meal data after pre-processing
meal_df = pd.read_csv('preprocessed_0.csv')
noMeal_df = pd.read_csv('preprocessed_1.csv')

############################################# FEATURES ####################################################################

def fn_zero_crossings(row, xAxis):
    slopes = [
     0]
    zero_cross = list()
    zero_crossing_rate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            zero_cross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    zero_crossing_rate = np.sum([np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(zero_cross) > 0:
        return [max(zero_cross)[0], zero_crossing_rate]
    else:
        return [
         0, 0]


def fn_mean_abs_change(row):
    mean = 0
    for i in range(0, len(row) - 1):
        mean = mean + np.abs(row[(i + 1)] - row[i])
    return mean / len(row)


def fn_rms(row):
    rms = 0
    for i in range(0, len(row) - 1):
        rms = rms + np.square(row[i])
    return np.sqrt(rms / len(row))


def fn_fft(row):
    FFT = fft(row)
    size = len(row)
    t = 0.006666666666666667
    frequency = np.linspace(0, size * t, size)
    amplitude = []
    phase = []
    for freq in FFT:
        amplitude.append(np.abs(freq))
        phase.append(math.atan2(np.imag(freq), np.real(freq)))

    sort_amplitude = amplitude
    sort_amplitude = sorted(sort_amplitude)
    max_amplitude = sort_amplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    max_phase = max(phase)
    return [
     max_amplitude, max_frequency, max_phase]


def fn_entropy(row):
    length = len(row)
    entropy = 0
    if length <= 1:
        return 0
    else:
        value, count = np.unique(row, return_counts=True)
        ratio = count / length
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy

########################################### FEATURE MATRIX ###############################################################
#Creating feature matrix by using the above functions

def features(df):
    df_features = pd.DataFrame()
    for i in range(0, df.shape[0]):
        row = df.iloc[i, :].tolist()
        df_features = df_features.append({'Mean_Abs_Change1':fn_mean_abs_change(row[:10]), 
         'Mean_Abs_Change2':fn_mean_abs_change(row[10:]), 
         'Skewness1':skew(row), 
         'Max_Zero_Crossing':fn_zero_crossings(row, df.shape[1])[0], 
         'Zero_Crossing_Rate':fn_zero_crossings(row, df.shape[1])[1], 
         'RMS1':fn_rms(row), 
         'Min1':min(row), 
         'Max1':max(row), 
         'Entropy1':fn_entropy(row), 
         'FFT_Max_Amplitude1':fn_fft(row[:10])[0], 
         'FFT_Max_Frequency1':fn_fft(row[:10])[1], 
         'FFT_Phase1':fn_fft(row[10:])[2], 
         'FFT_Max_Amplitude2':fn_fft(row[10:])[0], 
         'FFT_Max_Frequency2':fn_fft(row[10:])[1], 
         'FFT_Phase2':fn_fft(row[10:])[2]},
          ignore_index=True)

    return df_features

# Calculating meal and no-meal features
meal_features = features(meal_df)
noMeal_features = features(noMeal_df)

######################################## DATA STANDARDIZATION #############################################################

ss = StandardScaler()
meal_ss = ss.fit_transform(meal_features)
noMeal_ss = ss.fit_transform(noMeal_features)

############################################# PCA ########################################################################
#Performing PCA for dimensionality reduction and selecting the top 5 components for further analysis

pca = PCA(n_components=5)
pca.fit(meal_ss)


# eigen_values = pca.explained_variance_
# eigen_vectors = pca.components_
# eigen_vectors = eigen_vectors.T
# eigen_features_meal = meal_ss.dot(eigen_vectors)

with open('pca_meal.pkl', 'wb') as (file):
    pickle.dump(pca, file)
    
meal_pca = pd.DataFrame(pca.fit_transform(meal_ss)) # PCA for meal data
noMeal_pca = pd.DataFrame(pca.fit_transform(noMeal_ss)) #PCA for no-meal data

meal_pca['class'] = 1  #Label for meal class:1
noMeal_pca['class'] = 0 #Label for no-meal class:0

data = meal_pca.append(noMeal_pca)
data.index = [i for i in range(data.shape[0])]

X = data.iloc[:, :-1] # Features
Y = data.iloc[:, -1]  # Class Labels

score = []
accuracy_scores=[] #Accuracy
f1=[] #F1 scores

################################################ MODEL TRAINING ################################################################
def train_model(model):
    kfold = KFold(5, True, 1)
    for tr, tst in kfold.split(X, Y):
        X_train, X_test = X.iloc[tr], X.iloc[tst]
        Y_train, Y_test = Y.iloc[tr], Y.iloc[tst]
        
        #Adaboost
        if model.upper()=='ADABOOST':
            model = AdaBoostClassifier(random_state=1)
        
        #Random Forest
        elif model.upper()=='RF':
            model = RandomForestClassifier(n_estimators=5)
         
        #Support Vector Machine              
        elif model.upper()=='SVM':
            model = SVC(kernel='rbf',gamma='scale',degree=3) #kernal can be poly too
        
        elif model.upper()=='GPC':
            kernel = 1.0 * RBF(1)
            model = GaussianProcessClassifier(kernel=kernel, random_state=0)
        
        #Logistic Regression
        else:
            model = LogisticRegression()
        
        model.fit(X_train, Y_train)
        score.append(model.score(X_test, Y_test))
        
        y_predicted = model.predict(X_test)
        precision = precision_score(Y_test, y_predicted, average='binary')
        recall = recall_score(Y_test, y_predicted, average='binary')
        f1.append( 2 * precision * recall / (precision + recall))
    
        accuracy_scores.append(accuracy_score(Y_test, y_predicted))
    
        
    print('Avg Score:', (np.sum(score) / 5)*100)
    print('Avg Accuracy Score:', (np.sum(accuracy_scores) / 5)*100)
    print('Avg F1 Score:', (np.sum(f1) / 5)*100)
    
    modelname='model'+'_Model.pkl'
    with open(modelname, 'wb') as (file):
        pickle.dump(model, file)


train_model('GPC') #Input required model name 


##################################################### Using Cross Validation Score method #########################################

def cross_validation():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    classifier = SVC(kernel = 'poly',gamma='scale')
    classifier.fit(X_train, y_train)
    
    #cross-validation
    cv_score = cross_val_score(classifier, X_train, y_train, cv=15)
    print("Accuracy: {:20f}+/- {:20f}" .format(cv_score.mean(), cv_score.std() * 2))
    
    #prediction
    y_predicted = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predicted) #Confusion Matrix
    accuracy =accuracy_score(y_test, y_predicted)
    print("Accuracy: {:20f}" .format(accuracy))