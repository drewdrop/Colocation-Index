# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:07:43 2020

@author: LC
"""

import ot
import pandas as pd
from scipy import stats
import time
import numpy as np
import ot.gpu
import cupy
from collections import Counter


Datall = pd.read_csv('2013data.csv')

city=Datall['city']
cityall = Counter(city)#Count the number of enterprises

p1 = dict((key, value) for key, value in cityall.items() if value > 0)
citylist=list(p1.keys())

df=pd.DataFrame(columns=['year','city','indj','indk','iloc','numj','numk'])


# function version
def get_colocation_index(sic, n, reg):
    
    print("current sic:", sic)
    print("--- %s seconds ---" % (time.time() - start_time))  
    
    colocation_indexs = []
    

    # sic_location = cupy.array(Data.loc[Data['SIC2'] == sic, ['Latitude', 'Longitude']].to_numpy())
    sic_location = Data.loc[Data['SIC2'] == sic, ['Latitude', 'Longitude']]
    sic_n = len(sic_location)
    sic_density = cupy.ones((sic_n, ))/sic_n

    target_sics = np.asarray(SIC_List, dtype = int)
    index = np.where(target_sics==sic)
    target_sics = np.delete(target_sics, index)

    for target_sic in target_sics:
        # print("current target sic:", target_sic)
        
        wassersteins = []

        #target_sic_location = cupy.array(Data.loc[Data['SIC2'] == target_sic, ['Latitude', 'Longitude']].to_numpy())
        target_sic_location = Data.loc[Data['SIC2'] == target_sic, ['Latitude', 'Longitude']]
        target_sic_n = len(target_sic_location)  
        target_sic_density = cupy.ones((target_sic_n, ))/target_sic_n

        M = ot.gpu.dist(sic_location, target_sic_location, to_numpy = False)
        wasserstein = cupy.asnumpy(cupy.sum(ot.gpu.sinkhorn(sic_density, target_sic_density, M, reg, to_numpy = False) * M))
        wassersteins.append(wasserstein)

        for batch in range(0, n):                        #target_sic_location_temp = cupy.array(Data.sample(n = target_sic_n).loc[ : , ['Latitude', 'Longitude']].to_numpy())
            target_sic_location_temp = Data.sample(n = target_sic_n).loc[ : , ['Latitude', 'Longitude']]
            M_temp = ot.gpu.dist(sic_location, target_sic_location_temp, to_numpy = False)
            wasserstein = cupy.asnumpy(cupy.sum(ot.gpu.sinkhorn(sic_density, target_sic_density, M_temp, reg, to_numpy = False) * M_temp))
            wassersteins.append(wasserstein)
        
        
        
        colocation_index = (100 - stats.percentileofscore(wassersteins[1:], wassersteins[0], kind = "strict"))/100
        # colocation_index = sum( wassersteins[t] > wassersteins[0] for t in range(1, n + 1))/100
        colocation_indexs.append(colocation_index)
        
        print(target_sic,"--- %s seconds ---" % (time.time() - start_time))
        
    return colocation_indexs

for c in citylist:
    Data=Datall[Datall['city']==c]    
    
    ind=Data['SIC2']
    indall = Counter(ind)
    p2 = dict((key, value) for key, value in indall.items() if value > 0)#Industry and quantity of City C
    SIC_List=list(p2.keys())#Industy list
    SIC_num=list(p2.values())  
    # use apply function 
    # calculate the first batch of 9 SICs
    start_time = time.time()
    print(c)
    print(len(SIC_List))
    input_sics = pd.Series(SIC_List)[0:len(SIC_List)]
    outcomes = input_sics.apply(get_colocation_index, n = 1000, reg = 10^-3)
    print("--- %s seconds ---" % (time.time() - start_time))    
    
    # transfer the outputs into pd dataframe
    co_location_index = pd.DataFrame()
    
    for i in range(len(outcomes)):
        outcomes[i].insert(i, 1)
        temp = pd.DataFrame(outcomes[i]).transpose()
        co_location_index = co_location_index.append(temp)
    co_location_index.index = range(len(outcomes))
    
    for x in range(len(SIC_List)):
        for y in range(x+1,len(SIC_List)):
            df= df.append([{'year':2013,'city':c,'indj':SIC_List[x],'indk':SIC_List[y],'iloc':co_location_index.loc[x][y],'numj':SIC_num[x],'numk':SIC_num[y]}], ignore_index=True)
            df= df.append([{'year':2013,'city':c,'indj':SIC_List[y],'indk':SIC_List[x],'iloc':co_location_index.loc[y][x],'numj':SIC_num[y],'numk':SIC_num[x]}], ignore_index=True)


    df.to_csv('result.csv')