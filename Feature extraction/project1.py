
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lokeshwar
"""



from FeatureExtraction import featureExtraction


CGMSeriesLunchList=['CGMSeriesLunchPat1.csv','CGMSeriesLunchPat2.csv','CGMSeriesLunchPat3.csv','CGMSeriesLunchPat4.csv','CGMSeriesLunchPat5.csv']
CGMDatenumLunchList=['CGMDatenumLunchPat1.csv','CGMDatenumLunchPat2.csv','CGMDatenumLunchPat3.csv','CGMDatenumLunchPat4.csv','CGMDatenumLunchPat2.csv']

fe=featureExtraction()
fe.get_data(CGMSeriesLunchList,CGMDatenumLunchList)
fe.pre_processing()
fe.feature_matrix()
fe.pca()


    
    
    
