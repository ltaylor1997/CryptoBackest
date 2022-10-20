 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 23:56:30 2021

@author: Lucas
"""

import Signals as  sig
from ftx_client import ftx_client
import pandas as pd
import numpy as np



ftx = ftx_client(a_key, s_key)
class sort():
    #initialize the object
    def __init__(self,client):
        self.client = client
        
        self.pairs = ['BTC/USD','ETH/USD','LTC/USD','DOGE/USD']
        #self.weights = self.maxWeights()
        # print(self.baseList)
    #function to get     
    def __complieDF__(self,interval):
        #dataframe list
        dfs = []
        #tick list
        tick = []
        #for loop to get all of it
        for i in range(len(self.pairs)):
            #use the client to get the pprices
            x = self.client.get_historical_prices(market = self.pairs[i], 
                                           resolution = interval)['result']   
            
            #checks to make sure you get a good response
            
            #turns dictionary into dataframe
            df = pd.DataFrame(x)
            
            df.reset_index(drop = True, inplace = True)
            
            #checks to make sure it has proper length
            if len(df) > 975:
                #don't want usdc, it isn't volatile enough to work with and messes the following lines up
                
                #rename the columns to work with signals
                df.rename(columns = {"time":"date"}, inplace = True)
                #convert the data into float 32 for the models
                conv_dict = {'open':np.float32 ,'close':np.float32 ,'high':np.float32 ,'low':np.float32 }
                df = df.astype(conv_dict)
                #append the dataframes and ticks into the new list
                dfs.append(df)
                tick.append(self.pairs[i])
        #return the two lists
        return dfs,tick

    
    #function to return buy sell list
    def buySell(self,method = "SVM",weights = False,interval = 3600):
        self.buyList = []
        self.sellList = []
        #get the dataframes and tickers
        self.dataframes, self.tradable = self.__complieDF__(interval = interval)
        #get the dataframe to use for the spread
        index_spread = self.tradable.index('BTC/USD')
        btc_spread = self.tradable.index('ETH/USD')
        
        #empty list for accuracy scores
        w = []
        for i in range(len(self.dataframes)):
           #turn the spread df into an easy to call variable
           if self.pairs[i] == "BTC/USD":
               spread = self.dataframes[btc_spread]
           else:
               spread = self.dataframes[index_spread]
           
            #rename and change data for porcessing
           spread.rename(columns = {"time":"date"}, inplace = True)
           conv_dict = {'open':np.float32 ,'close':np.float32 ,'high':np.float32 ,'low':np.float32 }
           spread = spread.astype(conv_dict)
           #initialize the signal object
           obj = sig.complexStrat(self.dataframes[i], spread)
           #run the model
           x = obj.SVM()
           #get the last prediction
           binary = x[len(x)-1:]['signal'].values[0]
           #if else to append the tickers to the proper buy, sell list
           
           if binary == 0:
               self.sellList.append(self.pairs[i])
           elif binary == 1:
               self.buyList.append(self.pairs[i])
               
               #if weighting, append the accuracy score as well
               if weights == True:
                   w.append(self.weights[i])
            
                   
               else:
                   pass
        # make the weight list
        w = np.array(w)/sum(w)
        if weights == True:
            return self.sellList,self.buyList,w
        else:
            return self.sellList,self.buyList
    def maxWeights(self,interval = 3600):
        #get the dataframes and tickers
        self.dataframes, self.tradable = self.__complieDF__(interval = interval)
        #get the dataframe to use for the spread
        index_spread = self.tradable.index('BTC/USD')
        btc_spread = self.tradable.index('ETH/USD')
        cv = []
        #empty list for accuracy scores
        w = []
        for i in range(len(self.dataframes)):
           #turn the spread df into an easy to call variable
           if self.pairs[i] == "BTC/USD":
               spread = self.dataframes[btc_spread]
           else:
               spread = self.dataframes[index_spread]
           
            #rename and change data for porcessing
           spread.rename(columns = {"time":"date"}, inplace = True)
           conv_dict = {'open':np.float32 ,'close':np.float32 ,'high':np.float32 ,'low':np.float32 }
           spread = spread.astype(conv_dict)
           #initialize the signal object
           obj = sig.complexStrat(self.dataframes[i],spread)
           #run the model
           x = obj.SVM(back = True,cv = True)
           c = x[1]['CV']
           cv.append(c)
        cv = np.array(cv)
        cv = cv - .5                                                                
        cv = (cv - np.mean(cv))/np.std(cv)
        
        e = 1 / len(self.pairs)
        ew = np.ones(len(self.pairs)) * e
        mw = ew + cv * e
        mw = np.where(mw > 0, mw,0)
        
        return mw
    def HFTBuySell():
        pass  

print(sort(ftx).buySell())
