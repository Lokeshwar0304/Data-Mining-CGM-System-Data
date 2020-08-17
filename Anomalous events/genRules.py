# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:14:56 2020

@author: Lokeshwar
"""

#import ARM as arm
import glob
import pandas as pd
import numpy as np
import math

cgmSeriesLunchfile = 'CGMSeriesLunchPat*.csv'
insulinBolusLunchfile = 'InsulinBolusLunchPat*.csv'
    

frequent_df=pd.DataFrame() #most frequent sets
confidence_df=pd.DataFrame() #largest confidence rules
anomalous_df =pd.DataFrame() #Anomalous rules

ca_df=pd.DataFrame()

################################################# BINS ############################################################

def get_bins(cind):
    bins=[]
    for i in df.iloc[:,cind]:
        bins.append(math.ceil(i/10)-4 if not math.isnan(i) else 0)
    return bins

########################################### FREQUENT ITEM SETS ####################################################


def encode_units(x):
    print(x)
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def get_sets():  
    sets = arm_df.applymap(encode_units)
    sets=sets.replace(np.nan,0)
    return sets


def fn_generate_frequent_itemsets(records):
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    arm_tran = te.fit(records).transform(records)
    tran_df = pd.DataFrame(arm_tran, columns=te.columns_)
    itemsets = apriori(tran_df, min_support=0.0001, use_colnames=True)
    fis=[]
    for i in range(itemsets.shape[0]):
        if(len(list(itemsets.iloc[:,1][i])))==3:
            fis.append(list(itemsets.iloc[:,1][i]))
    return fis
    
    
########################################################################################################################
    
def fn_dfTolist(tdf):
    records = []
    for i in range(0, len(tdf)):
        records.append([str(arm_df.values[i,j]) for j in range(0,3)])
    return records
    
    
############################################## CONFIDENCE CALCULATION ###################################################
    
def fn_calConfidence(a_df):
    conf_list=[]
    anom_list=[]
    num_conf_df=pd.DataFrame(xx,columns=['max_cgm_bins','cgm_lunch_bins','max_insulinBolus'])
    num_conf_df['count']=0
    
    den_conf_df=num_conf_df.iloc[:,0:2]
    den_conf_df['count']=0
    
    numer=num_conf_df.groupby(['max_cgm_bins','cgm_lunch_bins','max_insulinBolus'])
    numer_count=numer.count()
    
    denom=den_conf_df.groupby(['max_cgm_bins','cgm_lunch_bins'])
    denom_count=denom.count()
    
    for i in range(0,numer_count.shape[0]):
        n=numer_count['count'][i]
        a=list(numer_count.index[i])
        for j in range(0,denom_count.shape[0]):
            b=list(denom_count.index[j])
            if(a[0:-1]==b):
                d=denom_count['count'][j]
                conf=(n/d)*100
                if(conf>=50):
                    conf_list.append([a,conf]) #largest confidence rules (Confidence ==100)
                else:
                    anom_list.append([a,conf]) # Anomalous rules (Confidence < 100)
                    
    return [conf_list,anom_list]
    
   
 
################################## FREQUENT ASSOCIATION RULES ##################################################### 
"""
Confidence Rules: Rules with confidence >= 50%
Anomalous  Rules: Rules with confidence < 50%

"""    


cscores=[]   # Confidence rule scores of all Confidence rules
ascores=[]   # Anomalous rule scores of all  Anomalous rules
    
for a,b in zip(glob.glob(cgmSeriesLunchfile),glob.glob(insulinBolusLunchfile)):
    cgm_df=pd.read_csv(a,index_col=False,header=0)
    bolus_df=pd.read_csv(b,index_col=False,header=0)
    
    df=pd.DataFrame()
    df['max_insulinBolus'] = bolus_df.max(axis=1)
    df['max_cgm'] = cgm_df.max(axis=1)
    df['max_cgm_bins']=get_bins(1)
    df['cgm_lunch'] = list(cgm_df.iloc[:,6])
    df['cgm_lunch_bins']=get_bins(3)
    
    arm_df=pd.DataFrame({'max_cgm_bins':df['max_cgm_bins'],
                     'cgm_lunch_bins':df['cgm_lunch_bins'],'max_insulinBolus':df['max_insulinBolus']})
    
    xx=fn_generate_frequent_itemsets(fn_dfTolist(arm_df))
    freqitems=[]
    for i in range(0,len(xx)):
        freqitems.append('('+ xx[i][0] + ',' + xx[i][1] + ',' + xx[i][2] + ')')
    frequent_df=frequent_df.append(freqitems, ignore_index=True)
     
    
    rules= fn_calConfidence(xx)
    
    # Confidence Rules
    conf_rules=[]
    
    for i in range(0,len(rules[0])):
        conf_rules.append('{'+ rules[0][i][0][0] + ', ' + rules[0][i][0][1] + '--> ' + rules[0][i][0][2] + '}')
        cscores.append(rules[0][i][1]) #Scores
    if(len(conf_rules)!=0):
        confidence_df=confidence_df.append(conf_rules, ignore_index=True)
    
    
    # Anomalous Rules
    an_rules=[]
    
    for i in range(0,len(rules[1])):
        an_rules.append('{'+ rules[1][i][0][0] + ', ' + rules[1][i][0][1] + '--> ' + rules[1][i][0][2] + '}')
        ascores.append(rules[1][i][1]) # Scores
    if(len(an_rules)!=0):
        anomalous_df=anomalous_df.append(an_rules, ignore_index=True)
    
########################################################## TO CSV ##################################################
        
frequent_df.to_csv('Frequent_Itemsets.csv',index=False)
confidence_df.to_csv('Confident_Rules.csv',index=False)
anomalous_df.to_csv('Anomalous_Rules.csv',index=False)



    
    
    
