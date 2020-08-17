# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:49:50 2020

Project 3

@author: Lokeshwar
"""
#C:\Users\Lokeshwar\Documents\MS\Data Mining\DataMiningProject3

import numpy as np
from scipy.stats import skew
import pandas as pd
from scipy.fftpack import fft
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob
import pickle
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from statistics import mode
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore")

mealDatafile = 'mealData*.csv'
mealAmountDatafile = 'mealAmountData*.csv'
    
#Read CSV and get first 50 rows only
df=pd.DataFrame()
for mdf,madf in zip(glob.glob(mealDatafile) ,glob.glob(mealAmountDatafile)):
    mdf_temp=pd.read_csv(mdf,names=[i for i in range(30)],index_col=False)
    madf_temp=pd.read_csv(madf,names=[30],index_col=False)
    concat_df=pd.concat([mdf_temp.iloc[:50,:],madf_temp.iloc[:50,:]],axis=1,sort=False) #get first 50 rows and concat
    df=pd.concat([df,concat_df],axis=0,sort=True)

df.index = [i for i in range(df.shape[0])] 
df.to_csv('concatDF.csv', index=False)  #Save the csv file

# Dropping rows where the number of NaN values are >=10
mrow_index = []
for row in range(0, df.shape[0]):
    if df.iloc[row, :].isnull().sum() >=10: 
        mrow_index.append(row)
df=df.drop(index=mrow_index) # dropping NaN values
df.index = [i for i in range(df.shape[0])] 

# Replacing remaining NaN values
df=df.interpolate(axis=1,method='quadratic',limit=10, limit_direction='both') 
df=df.dropna(axis=0,how='any')
df.index = [i for i in range(df.shape[0])] 


############################################# FEATURES ####################################################################
    
def slopes(row,xAxis):
    slopes = []
    X = [i for i in range(xAxis)]
    Y = list(row)
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))
    return(sum(slopes[0:6])/30,sum(slopes[6:12])/30,sum(slopes[12:18])/30)


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
    
def fn_timeDiff(row): # New Feature
    max_cgm=max(row)
    return 0 if row.index(max_cgm)*5- 30<0 else row.index(max_cgm)*5- 30

##############################################################################################################

def allocate_bins(): #Allocating bins by considering skewness in the total number of bins
    leo=[]
    for i in range(0,6,1):
        li=df[df['dbscan_clusters']==i]['bins']
        b=[]
        for i in range(0,6,1):
            b.append(list(li).count(i))
        leo.append(b)
        
    binc=[]
    for i in range(0,6,1):
        binc.append(list(df['bins']).count(i))
        
    kbc=pd.DataFrame({'bin_count':binc,'C0':leo[0],
                                             'C1':leo[1],
                                             'C2':leo[2],
                                             'C3':leo[3],
                                             'C4':leo[4],
                                             'C5':leo[5]})
            
    ratio=[9,7.428,4.428,8.428,3,1] # Bias estimation of different bins wrt cluster 5
    nr=[]
    for i in range(0,6,1):
        row=kbc.iloc[i,1:]
        nr1=[]
        for r in row:
            nr1.append(round(r/ratio[i],2))  
        nr.append(nr1)
    
    kbc4=pd.DataFrame(nr,columns=['C0','C1','C2','C3','C4','C5'])
    

############################################## FEATURE MATRIX ######################################################
        

def features(df):
    df_features = pd.DataFrame()
    for i in range(0, df.shape[0]):
        row = df.iloc[i, :].tolist()
        df_features = df_features.append({
         'Mean_Abs_Change':fn_mean_abs_change(row), 
         'Skewness':skew(row), 
         'RMS':fn_rms(row), 
         'Min':min(row), 
         'Max':max(row),
         'CGMdiff1':(max(row)-row[6])/row[6] if row[6]!=0 else 0, # max(CGM) - CGM at T=6 / CGM at T=6 (New Feature)
         'Twidth':fn_timeDiff(row),
         'Variance':np.var(row),
         'InsulinRise':row[9]-row[6]/(45-30),
         'Slope1':slopes(row,df.shape[1]-1)[0],
         'Slope2':slopes(row,df.shape[1]-1)[1],
         'Slope3':slopes(row,df.shape[1]-1)[2],
         'Entropy':fn_entropy(row), 
         'FFT_Max_Amplitude':fn_fft(row)[0], 
         'FFT_Max_Frequency':fn_fft(row)[1], 
         'FFT_Phase':fn_fft(row)[2]},
          ignore_index=True)

    return df_features

meal_features = features(df.iloc[:,:-1]) # All columns except the last one


######################################## DATA STANDARDIZATION #############################################################

ss = StandardScaler()
meal_ss = ss.fit_transform(meal_features)

############################################# PCA and MODEL TRAINING #####################################################

pca = PCA(n_components=2)
pca.fit(meal_ss)

with open('pca_meal.pkl', 'wb') as (file):
    pickle.dump(pca, file)
    
meal_pca = pd.DataFrame(pca.fit_transform(meal_ss))
meal_pca.to_csv('train_pca')

#########################################  K MEANS ##################################################################

kmeans = KMeans(n_clusters=6,max_iter=1000, algorithm = 'auto')
cluster_labels=kmeans.fit_predict(meal_pca)
pLabels=list(kmeans.labels_)
#print("KMEANS SSE: ", kmeans.inertia_)

#######################################################################################################################
bins=[]
for i in df.iloc[:,30]:
    bins.append(1 if (i==0) else 2 if (i>=1 and i<=20) else 3 if(i>=21 and i<=40) else 4 if(i>=41 and i<=60) 
    else 5 if(i>=61 and i<=80) else 6 if(i>=81 and i<=100) else 7)
df['bins']=bins
df['kmeans_clusters']=pLabels #predicted labels/clusters
df['kmeans_bins']=df['kmeans_clusters']
df['kmeans_bins']=df['kmeans_bins'].map({0:6,1:4,2:2,3:3,4:5,5:1})
#print("The average silhouette_score for KMEANS is :", silhouette_score(meal_pca, cluster_labels))

######################################### PLOTTING KMEANS #########################################################

def plot_kmeans():
    
    centers = kmeans.cluster_centers_
    
    plt.figure()
    plt.scatter(meal_pca.iloc[:, 0], meal_pca.iloc[:, 1], c=cluster_labels, cmap="plasma",alpha=1)
    plt.scatter(centers[:, 0],   
                centers[:, 1],
                marker='^', 
                c=[0, 1, 2, 3, 4,5], 
                s=100, 
                linewidth=2,
                cmap="plasma")
    plt.show()


###################################### DBSCAN #######################################################################
#The optimal values of eps and min_samples are selected from a trail run of different combinations

eps=0.49999999999999994
min_sample=7
db = DBSCAN(eps=eps, min_samples=min_sample)
clusters = db.fit_predict(meal_pca)
dbscan_df=pd.DataFrame({'pc1':list(meal_pca.iloc[:,0]),'pc2':list(meal_pca.iloc[:,1]),'dbscan_clusters':list(clusters)})
outliers_df=dbscan_df[dbscan_df['dbscan_clusters']==-1].iloc[:,0:2]


#######################################################################################################################

# Two ways to merge DBScan outliers into its nearest neighbour clusters.
# 1. Using KNN
# 2. Using Bisecting KMeans

####################################### KNN FOR OUTLIERS ##############################################################
#Using KNN, the outliers in the DBScan are merged into its nearest clusters. 

knn = KNeighborsClassifier(n_neighbors=4,p=2)
knn.fit(dbscan_df[dbscan_df['dbscan_clusters']!=-1].iloc[:,0:2],dbscan_df[dbscan_df['dbscan_clusters']!=-1].iloc[:,2])
for x,y in zip(outliers_df.iloc[:,0],outliers_df.iloc[:,1]):
    dbscan_df.loc[(dbscan_df['pc1'] == x) & (dbscan_df['pc2'] == y),'dbscan_clusters']=knn.predict([[x,y]])[0]

####################################################PLOT DBSCAN #########################################################
def plot_dbscan():
    
    fig, axs = plt.subplots(1, 2,squeeze=False)
    axs[0, 0].scatter(meal_pca.iloc[:,0], meal_pca.iloc[:,1],c=clusters, cmap='Paired') 
    axs[0, 1].scatter(dbscan_df.iloc[:,0], dbscan_df.iloc[:,1],c=dbscan_df.iloc[:,2], cmap='Paired')

############################################### BISECTING KMEANS #########################################################

largestClusterLabel=mode(dbscan_df['dbscan_clusters'])
biCluster_df=dbscan_df[dbscan_df['dbscan_clusters']==mode(dbscan_df['dbscan_clusters'])]
bi_kmeans = KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(biCluster_df)
bi_centroids = bi_kmeans.cluster_centers_
bi_pLabels=list(bi_kmeans.labels_)
biCluster_df['bi_pcluster']=bi_pLabels
biCluster_df=biCluster_df.replace(to_replace =0,  value =largestClusterLabel) 
biCluster_df=biCluster_df.replace(to_replace =1,  value =max(dbscan_df['dbscan_clusters'])+1) 

for x,y in zip(biCluster_df['pc1'],biCluster_df['pc2']):
   newLabel=biCluster_df.loc[(biCluster_df['pc1'] == x) & (biCluster_df['pc2'] == y)]
   dbscan_df.loc[(dbscan_df['pc1'] == x) & (dbscan_df['pc2'] == y),'dbscan_clusters']=newLabel['bi_pcluster']
df['dbscan_clusters']=dbscan_df['dbscan_clusters']
dbscan_df['dbscan_bins']=dbscan_df['dbscan_clusters']
dbscan_df['dbscan_bins']=dbscan_df['dbscan_bins'].map({0:1,1:6,2:2,3:5,4:3,5:4})
print("KMEANS and DBSCAN created successfully and respective bins are assigned to clusters.")

##########################################################################################################################

train_features=meal_features
train_features['bins']=bins
train_features['kmeans_bins']=df['kmeans_bins']
train_features['dbscan_bins']=dbscan_df['dbscan_bins']
meal_features.to_csv('train_features.csv',index=False)

###########################################################################################################################
