#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 09:00:05 2021

@author: Lucas
"""

import numpy as np
import pandas as pd
import talib
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore') 

class strat():
    """
    Class to get a signal from price data. Only requires you to 
    put in a dataframe with historical data including the columns 
    -Open
    -High 
    -Low
    -Close
    -Volume
    In addition, the data needs to go in chronological order
    """
    #construct the object
    def __init__(self,USDPrices,coinPrices = None):
        self.USDPrices = USDPrices.rename(columns = str.lower)
        self.coinPrices =  coinPrices.rename(columns = str.lower)
    #MACD function
    def MACD(self, shortWindow = 12, longWindow = 26, back = False):
        
        #get the prices and only use close for ease of running
        P = self.USDPrices['close']
        #calculate the MACD
        MACD = talib.MACD(P ,shortWindow,longWindow)
        #concatenate prices and MACD
        P = pd.concat([P,MACD[0]],axis = 1)
        #change the column names
        P.columns = ['close','MACD']
        #turn the MACD into simple buy sell signal
        P = P.fillna(0)
        P['MACDsignal'] = np.where(P['MACD'] > 0 , 1 , 0 )
        #return close prices, indicator, signal
        
        backtest = self.metrics(P,'MACDsignal')
        if back == True:
            return P, backtest
        else:
            return P
    #function for trading the spread between different coins
    def spread(self,back = False):
        #get the coin prices in USD
        P = self.USDPrices
        #get the coin to coin prices, i.e. ETH/BTC
        S = self.coinPrices['close']
        #calculate the MACD
        #could use this for correlation or something different
        #open to change will program this later
        MACD = talib.MACD(S)
        #turn the spread into a signal
        P['Spread MACD'] = MACD[0]
        P['SpreadSignal'] = np.where(MACD[0] > 0,1,0)
        #return prices, indicator, signal
        backtest = self.metrics(P,'SpreadSignal')
        P = P.fillna(0)
        if back == True:
            return P, backtest
        else:
            return P
    
    #bollinger bands 
    def BBands(self,method = "touch",back = False):
        #get coin prices in USD
       P = self.USDPrices
       #one std bands
       x1 = talib.BBANDS(P['close'], nbdevup = 1,nbdevdn =1)
       #two std bands
       x2 = talib.BBANDS(P['close'], nbdevup = 2,nbdevdn =2)
       #concat the bands into 1 df and change column names for each
       combine1 = pd.concat(x1,axis = 1)
       combine1.columns = ['upper1','middle1','lower1']
       combine2 = pd.concat(x2,axis = 1)
       combine2.columns = ['upper2','middle2','lower2']
       #put it all together with prices
       P = pd.concat([P,combine1,combine2],axis = 1)
       #initialize the signal column
       P['BBandsSignal'] = np.zeros(len(P))
       #if the price touchs the top band, sell, and vice versa lower band
       if method == "touch":
           #for loop, change it row by row if the close interacts with the band
           for i in range(len(P)):
               if P['close'][i] >= P['upper1'][i]:
                   P['BBandsSignal'][i] = 0
               elif P['close'][i] <= P['lower1'][i]:
                   P['BBandsSignal'][i] = 1
       #if the price touches top band buy, and vice versa lower band
       elif method == "trend":
            #for loop, change it row by row if the close interacts with the band
            for i in range(len(P)):
               if P['close'][i] >= P['upper1'][i]:
                   P['BBandsSignal'][i] = 1
               elif P['close'][i] <= P['upper1'][i]:
                   P['BBandsSignal'][i] = 0
       backtest = self.metrics(P,'BBandsSignal')
       P['bband%'] = (P['close'] - P['lower1'])/(P['upper1'] - P['lower1'])
       P = P.fillna(0)
       if back == True:
            return P, backtest
       else:
            return P
   #sees if lag returns can be a predictor for 
    def SimpLagReturns(self,period,method = "trend",back = False):
        P = self.USDPrices 
        P['returns'] = np.log(P['close']/P['close'].shift(1))
        P['Lag Returns'] = P['returns'].shift(period)
        P['LagSignal'] = np.zeros(len(P))
        P = P.fillna(0)
        for i in range(len(P)):
            if method == "trend":
                if P['Lag Returns'][i] > 0:
                    P['LagSignal'][i] = 1
            elif method == "reversion":
                if P['Lag Returns'][i] < 0:
                    P['LagSignal'][i] = 1
        backtest = self.metrics(P,'LagSignal')
        
        if back == True:
            return P, backtest
        else:
            return P
    
    def trange(self,back = False):
        P = self.USDPrices
        Trange = talib.TRANGE(P['high'],P['low'],P['close'])
        P['true range'] = Trange.fillna(0)
        P['Trange signal'] = np.where(P['true range'] > P['true range'].shift(1),1,0)
        P['trange%'] = np.log(P['true range']/P['true range'].shift(1))
        performance = self.metrics(P,'Trange signal')
        if back == True:
            return P,performance
        elif back == False:
            return P
    
    def metrics(self,strategy,name):
        strategy['returns'] = np.log(strategy['close']/strategy['close'].shift(1))
        strategy['FwdReturns'] = strategy['returns'].shift(-1)
        df = strategy.fillna(0)
        r = df['FwdReturns'].values
        s = df[name].values
        rB = np.where(r> 0,1,0)
        accuracy = accuracy_score(rB, s)
        strat_r = (r @ s)
        ones = np.ones(len(r))
        coin_r = (r @ ones)
        alpha = strat_r - coin_r
        stdR = r.std()
        r = np.array_split(r,10)
        ones = np.array_split(ones,10)
        
        s = np.array_split(s,10)
        periodR = []
        periodC = []
        for i in range(len(r)):
            rP = (r[i] @ s[i])
            rC = (r[i] @ ones[i])
            periodR.append(rP)
            periodC.append(rC)
        R = np.array(periodR)
        C = np.array(periodC)
        periodA = (R-C)
        edge = (max(R) - abs(min(R)))/(max(R) + abs(min(R)))
        performance = {"Accuracy":accuracy,"alpha":alpha,"PeriodByPeriodAlpha": periodA,"returns":strat_r,"returnsBtPeriod":R,'Edge': edge,"retrunStd": stdR}
        return performance
    
class stratCombine(strat):
    """
    Class to combine all of the simple strategies into one dataframe
    Need to pass in the coin prics in USD and coin prices in another coin
    
    """
    #inherant the initializatino from parent class
    def __init__(self,USDPrices,coinPrices):
        super().__init__(USDPrices,coinPrices  )
        #self.binary = binary
    def signalCreateAll(self):
        """
        puts all of the technical indicators together in one dataframe
        """
        #line to make sure there are no extra columns in the dataframe
        df = self.USDPrices[['date','open','low','high','close','volume']]
        #create all of the simple strategies
        #get both the binary and continous signals
        MACD = super(stratCombine,self).MACD()[['MACDsignal','MACD']]
        BBands = super(stratCombine,self).BBands()[['BBandsSignal','bband%']]
        spread = super(stratCombine,self).spread()[['SpreadSignal','Spread MACD']]
        SimpLagReturns = super(stratCombine,self).SimpLagReturns(period = 1)[['LagSignal','Lag Returns']]
        trange = super(stratCombine, self).trange()[['true range','Trange signal']]
        #put it all into one dataframe
        P = pd.concat([df,MACD,BBands,spread,SimpLagReturns,trange], axis = 1)
        
        return P
    #function to automatically filter the bad signals
    def signalFilter(self):
        MACD = super(stratCombine,self).MACD(back = True)
        BBands = super(stratCombine,self).BBands(back = True)
        spread = super(stratCombine,self).spread(bacl = True)
        SimpLagReturns = super(stratCombine,self).SimpLagReturns(period = 1, back = True)


class complexStrat(stratCombine):
    def __init__(self,USDPrices,coinPrices):
        super().__init__(USDPrices,coinPrices)
    def f_score(self,filterLevel = 3,back = False):
        """
        Sorting by technical indicators 
        filterLevel: how many technical indicators have to occur before a buy signal
        
        """
        #get all of the compiled signals
        signal = super(complexStrat,self).signalCreateAll()
        #calculate the f score
        signal['F score'] = signal['MACDsignal'] + signal['BBandsSignal']  + signal['SpreadSignal'] + signal['LagSignal']        
        #turn it into a signal 
        signal['F Score Signal'] = np.where(signal['F score'] > filterLevel,1,0)
        #fill the na as 0, don't want to buy if we don't have information
        signal = signal.fillna(0)
        #perform the metrics compilation
        performance = super(complexStrat,self).metrics(signal,'F Score Signal')
        #if back is false, just returns dataframe
        if back == False:
            return signal
        #else, returns performance metrics and fataframe
        elif back == True:
            return signal,performance
        
    def logistic(self,binary = False,back = False):
        """
        returns dataframe of signals and indicators
        back: controls if it returns backtest metrics or not
        
        """
        #put the scaler and regression for ease of running
        scaler = StandardScaler()
        model = LogisticRegression()
        model2 = PCA(n_components = 4)
        #get the dataframe of all the indicators
        signal = super(complexStrat,self).signalCreateAll()
        # calculate returns and forward returns
        signal['returns'] = np.log(signal['close']/signal['close'].shift(1))
        signal['Fwd returns'] = signal['returns'].shift(-1)
        #get just the indicators for data running
        #if binary is true, only uses th signals, otherwise it is continuous
        if binary ==True:
            X = signal[['MACDsignal','SpreadSignal','BBandsSignal','LagSignal','Trange signal']]
        #continuous variables
        elif binary == False:
            
            X = signal[['MACD','bband%','Spread MACD','Lag Returns' ]].fillna(0)
            #transform independent variables
            X = scaler.fit_transform(X)
            X = model2.fit_transform(X)
            #get the y variable and make it binary
        y = np.where(signal['Fwd returns'].fillna(0) > 0,1,0)
        #split into training adn test data
        X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.20, random_state = 7 )
        #fit the model and predict
        model.fit(X_train,y_train)
        #predict and turn into a signal
        predictions = model.predict(X)
        signal['logistic'] = predictions
        #get performance metrics
        performance = super(complexStrat,self).metrics(signal,'logistic')
        #control whether metrics are returned
        if back == False:
            return signal
        elif back ==True:
            return signal, performance
    #def XGBoost(self):
        
        
    