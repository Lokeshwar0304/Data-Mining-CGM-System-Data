# Data Mining: Continuous Glucose Monitoring System Timeseries Data
## (Time Series Analysis to Predict meal intake of diabetic patients)
<!-- TABLE OF CONTENTS -->
## Table of Contents

  * [About the Project](#about-the-project)
    * [Data Set](#data-set)
    * [Built With](#built-with)
    * [Phase vs Folder](#phase-vs-folder)
  * [Feature Extraction](#feature-extraction)
  * [Phase 2](#phase-2)
  * [Phase 3](#phase-3)
  * [Phase 4](#phase-4)
  * [Learning Outcome](#learning-outcome)

<!-- ABOUT THE PROJECT -->
## About The Project
The project is about extracting time-series features from CGM(Continuous Glucose Monitoring) System data, classifying whether a patient had a meal or not, clustering meal data  and detecting anomalous events.

### Data Set
The dataset is a close dataset that consists of following variables which are collected over a lunch meal for a period of 2.5 hours with an interval of 5 minutes from a CGM system.</br>
* Glucose Levels
* Time stamps of the glucose taken
* Insulin Basal Infusion
* Insulin Bolus Infusion
* Time stamps for each Basal or Bolus Infusion

### Built With
* Python
* Spyder
* Anaconda
### Phase vs Folder
* Phase 1 - Feature Extraction
* Phase 2 - Classifying meal or no-meal
* Phase 3 - Meal data clusters
* Phase 4 - Anomalous events

## Feature Extraction
Multiple *time-series features* were extracted from the CGM system data by measuring the frequency, magnitude, and fluctuations using statistical analysis techniques.
Extracted Features:</br>
* Fast Fourier Transform(Amplitude, Frequency, Phase)
* Zero Crossings
* Poly Fit
* Mean Absolute Change
* Windowed Mean</br>

Feature matrix is created and PCA was applied to reduce the dimensions of the data set and select only top k components

## Phase 2
The goal of this phase is to determine whether a person had meal or not. <br/>
Implemented *Adaboost, Random Forest, Gaussian Process Classifier, and SVM* using the extracted feature sets, compared using the ROC curve and selected Gaussian Process Classifier **(F1 score: 81%)**. <br/>
Given an input, the model will determine whether the person had meal or not. <br/>
Below are the F1 scores and accuracy of the implemented classification algorithms.</br>

| Classifer | F1 Score |
| :---         |            ---: |
| Adaboost   | 0.68    |
| Random Forest    | 0.63      |
| Gaussian Process Classifier    |  0.81    |
| Support Vector Machine    |  0.77     |
| Logistic Regression    |  0.51      |

## Phase 3
The goal of this phase is to cluster meal data based on the amount of carbohydrates in each meal. <br/>
Apart from the features already extracted, few more features were added in this phase.<br/>
Added Features:
* Insulin rise between 30 and 45 minutes
* CGM difference, which is the difference between maximum CGM for the day and CGM at 30 minutes.
* Time difference at which the maximum CGM for the day is observed and 30<br/>

Implemented *KMeans and DBScan* algorithms to find clusters<br/>
The silhouette_score for KMeans is : **0.70343**</br>
*K-Nearest Neighbours and Bisecting KMeans* were used to assign the outliers from the DBScan to its nearest clusters.</br>
Given the test input, the data points can be assigned to its respective cluster using trained KMeans or DBScan model.

## Phase 4
The aim of this phase is to determine anomalous events through *Association rule mining*.  
Features used in this phase are maximum bolus insulin level, maximum CGM, CGM value during the start of the lunch.  
Max CGM values and CGM values during lunch were quantized into bins respectively.  
The goal is to find most frequent itemsets for each feature.  
Extract all the rules, calculate confidence of each observed rule, find rule with the largest confidence for each feature.    
The least confidence rules are tagged as anomalous events.  
