#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lokeshwar
"""

import numpy as np
from numpy.polynomial import Polynomial as polynomial
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob


class featureExtraction:
    def __init(self):
        self.df=pd.DataFrame()
        self.df1=pd.DataFrame()
        self.df_features=pd.DataFrame()
        self.df_CGMSeries=pd.DataFrame()
        self.df_CGMDatenum=pd.DataFrame()


    def get_data(self, CGMSeriesLunchList,CGMDatenumLunchList):  
        #CGMDatenumLunchList=['CGMDatenumLunchPat1.csv','CGMDatenumLunchPat2.csv','CGMDatenumLunchPat3.csv','CGMDatenumLunchPat4.csv','CGMDatenumLunchPat2.csv'] 
        #Reading the CSV files and creating dataFrame
        self.df=pd.concat([pd.read_csv(name) for name in CGMSeriesLunchList])
        self.df1=pd.concat([pd.read_csv(name) for name in CGMDatenumLunchList])

##################################   DATA PREPROCESSING   ################################################

    def pre_processing(self):
        
        # Dropping all the rows where the count of Null/NaN > 7 in a  particular row
        row_index=[]
        for row in range(0,self.df.shape[0]): 
            if(self.df.iloc[row,:].isnull().sum()>7):
                row_index.append(row)
        
        # Dropping columns where the count of Null/NaN > 20 in a particular column
        column_index=[]
        for column in range(0,self.df.shape[1]):
            if(self.df.iloc[:,column].isnull().sum()>20):
                column_index.append(column)
        
        self.df_CGMSeries=self.df.drop(index=row_index,columns=[self.df.columns[c] for c in column_index])  
        self.df_CGMDatenum=self.df1.drop(index=row_index,columns=[self.df1.columns[c] for c in column_index])  
        self.df_features=pd.DataFrame()
        
        #For each row using poly fit, filling the NaN values and using the coefficients as features
        self.PolyFit_Coeff_A=[]
        self.PolyFit_Coeff_B=[]
        self.PolyFit_Coeff_C=[]
        self.PolyFit_Coeff_D=[]
        self.PolyFit_Coeff_E=[]
        
        for row in range(0,self.df_CGMSeries.shape[0]):
             X=self.df_CGMDatenum.iloc[row,:][::-1].reset_index().drop(['index'],axis=1)
             Y=self.df_CGMSeries.iloc[row,:][::-1].reset_index().drop(['index'],axis=1)
             
             #plt.plot(X,Y,label='Original Curve',color='b')
             
             P = polynomial.fit(np.array(X[~Y.isnull().any(axis=1)].iloc[:,0]),np.array(Y[~Y.isnull().any(axis=1)].iloc[:,0]),4)
             Y[Y.isnull().any(axis=1)]=P(X[Y.isnull().any(axis=1)])
             self.PolyFit_Coeff_A.append(list(P)[0])
             self.PolyFit_Coeff_B.append(list(P)[1])
             self.PolyFit_Coeff_C.append(list(P)[2])
             self.PolyFit_Coeff_D.append(list(P)[3])
             self.PolyFit_Coeff_E.append(list(P)[4])
             
             self.df_CGMSeries.iloc[row,:]=(Y.iloc[:,0][::-1]).tolist()
        
        #Preprocessed csv
        #self.df_CGMSeries.to_csv("PreProcessedData.csv") 
########################################### FEATURES ######################################################

########################################### FFT #################################################

    def fn_fft(self,row):
        print('in fft')
        FFT=fft(self.df_CGMSeries.iloc[row,:]) 
        size=len(self.df_CGMSeries.iloc[row,:]) 
        sample_interval=self.df_CGMDatenum.iloc[row,0]-self.df_CGMDatenum.iloc[row,1]
        frequency=np.linspace(0,1/sample_interval,size) #frequency is inverse of time
    
        amplitude=[]
        phase=[]
        for freq in FFT:
            amplitude.append(np.abs(freq)) # Calculate amplitude which is absolute value of the frequency component (Complex number)
            phase.append(math.atan2(np.imag(freq),np.real(freq))) # in radians 
       
        sort_amplitude=amplitude
        sort_amplitude=sorted(sort_amplitude)
        # Taking second max amplitude and its respecive frequency value
        max_amplitude=sort_amplitude[-2]
        min_amplitude=sort_amplitude[0]
        max_frequency=frequency.tolist()[amplitude.index(max_amplitude)]
        min_frequency=frequency.tolist()[amplitude.index(min_amplitude)]
        max_phase= max(phase)
        # Returning max and min amplitude, its frequencies and phase
        return ([max_amplitude,min_amplitude,max_frequency,min_frequency,(max_frequency-min_frequency),max_phase]) 


########################################### ZERO CROSSINGS #################################################

    def fn_zero_crossings(self,row):
        print('in zc')
        slopes=[0]
        zero_cross=list()
        zero_crossing_rate=0
        X=self.df_CGMDatenum.iloc[row,:][::-1].tolist()
        Y=self.df_CGMSeries.iloc[row,:][::-1].tolist()
        for index in range(0,len(X)-1):
            slopes.append((Y[index+1]-Y[index]) / (X[index+1]-X[index]))
        
        for index in range(0,len(slopes)-1):
            if(slopes[index]*slopes[index+1]<0):
                zero_cross.append([slopes[index+1]-slopes[index],X[index+1]])
        zero_crossing_rate= (np.sum([np.abs(np.sign(slopes[i+1])-np.sign(slopes[i])) for i in range(0,len(slopes)-1)]))/ (2* len(slopes))
        
        
        if(len(zero_cross)>0):
            return([len(zero_cross),max(zero_cross)[0],zero_crossing_rate])
        else:
            return ([0,0,0])

##############################################################################################################
 
    def fn_mean_abs_change(self,row):
        print('in mbc')
        mean=0
        sample=self.df_CGMSeries.iloc[row,:].tolist()
        for i in range(0,len(sample)-1):
            mean=mean+ np.abs(sample[i+1]-sample[i])
        return (mean/len(sample))

####################################### FEATURE MATRIX ##########################################################
    def feature_matrix(self):
        print('in fm')
        for row in range(0,self.df_CGMSeries.shape[0]):
            self.df_features=self.df_features.append({
                    'Poly_Coeff_A':self.PolyFit_Coeff_A[row],
                    'Poly_Coeff_B':self.PolyFit_Coeff_B[row],
                    'Poly_Coeff_C':self.PolyFit_Coeff_C[row],
                    'Poly_Coeff_D':self.PolyFit_Coeff_D[row],
                    'Poly_Coeff_E':self.PolyFit_Coeff_E[row],
                    'Mean_Abs_Change':self.fn_mean_abs_change(row), 
                    'Skewness':skew(self.df_CGMSeries.iloc[row,:]),
                    'FFT_Max_Amplitude':self.fn_fft(row)[0],
                    'FFT_Min_Amplitude':self.fn_fft(row)[1],
                    'FFT_Max_Frequency':self.fn_fft(row)[2],
                    'FFT_Min_Frequency':self.fn_fft(row)[3],
                    'FFT_Freq_Diff':self.fn_fft(row)[4],
                    'FFT_Phase':self.fn_fft(row)[5],
                    'Zero_Crossings': self.fn_zero_crossings(row)[0],
                    'Max_Zero_Crossing': self.fn_zero_crossings(row)[1],
                    'Zero_Crossing_Rate': self.fn_zero_crossings(row)[2]
                      },ignore_index=True)
         
        #Write the feature matrix to CSV
        #self.df_features.to_csv("Feature_Matrix.csv")

######################################## P C A #############################################################

    def pca(self):
        cgm_ss = StandardScaler().fit_transform(self.df_features) # features standardization
        #print(self.df_pca.shape)
        #np.std(self.df_pca)
        pca = PCA()
        cgm_components = pca.fit_transform(cgm_ss)
        self.df_principal = pd.DataFrame(data=cgm_components)
        
        eigen_values=pca.explained_variance_
        eigen_vectors=pca.components_
        
        #Since we need to consider only top 5 components
        #self.df_principal.iloc[:,0:4].to_csv("Principal_Matrix.csv") 
        #pd.DataFrame(eigen_vectors).to_csv("Eigen_Vectors.csv")
        var_total=np.sum(eigen_values)  
        for i in range(0,5):
            print('Principal Component '+str(i+1) + ' holds '+ str((eigen_values[i]/np.sum(var_total))*100) + ' % of the information.') 


####################################### PLOTTING FEATURES ##########################################################
# =============================================================================
# #1.SKEWNESS
# sns.distplot(np.array(self.df_features['Skewness']),bins=50,kde=True,color='b')
# plt.axvline(x=np.mean(self.df_features['Skewness']),color='r',linewidth=4)        
# =============================================================================
#2 FFT
# plt.bar(frequency[:size // 2], np.abs(FFT)[: size// 2] * 1 / size,linewidth=1,width=1.5,color='r')
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.show()
# =============================================================================

#   plt.cla()
#   plt.plot(slopes)
#   plt.axhline(y=0,color='black',linewidth=0.8)
#   plt.axvline(x=diff_slopes[1],color='r',linewidth=2)
# =============================================================================

# #Plotting random rows
# for row in range(1,10,2):
#   fig, ax = plt.subplots()
#   ax.plot(self.df_CGMDatenum.iloc[row,:].tolist(), self.df_CGMSeries.iloc[row,:].tolist())
#   ax.set(xlabel='Time', ylabel='Glucose levels')
#   ax.grid()
#   plt.show()


# =============================================================================
#plt.cla()
#plt.title('Zero_Crossings',fontsize=25)
#plt.xlabel('Time Series',fontsize=25)
#plt.ylabel(' Feature Values',fontsize=25)
#plt.plot(self.df_features['Zero_Crossings'],color='b',linewidth=4)  
# =============================================================================

    
    
