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
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn import svm
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
        P = self.USDPrices
        #calculate the MACD
        #MACD = talib.MACD(P ,shortWindow,longWindow)
        #concatenate prices and MACD
        P['12 day EWA'] = P['close'].ewm(span = 12).mean()
        P['26 day EWA'] = P['close'].ewm(span = 26).mean()
        P['MACD'] = P['12 day EWA'] - P['26 day EWA']
        
        #P = pd.concat([P,MACD[0]],axis = 1)
        #change the column names
        #P.columns = ['close','MACD']
        #turn the MACD into simple buy sell signal
        P = P.fillna(0)
        P['MACDsignal'] = np.where(P['MACD'] > 0 , 1 , 0 )
        #return close prices, indicator, signal
        
        
        if back == True:
            backtest = self.metrics(P,'MACDsignal')
            return P, backtest
        else:
            return P
    #function for trading the spread between different coins
    def spread(self,back = False):
        #get the coin prices in USD
        P = self.USDPrices
        #get the coin to coin prices, i.e. ETH/BTC
        S = self.coinPrices
        #calculate the MACD
        #could use this for correlation or something different
        #open to change will program this later
        S['close'] = S['close']/P['close']
        S['12 day EWA'] = S['close'].ewm(span = 12).mean()
        S['26 day EWA'] = S['close'].ewm(span = 26).mean()
        S['Spread MACD'] = S['12 day EWA'] - S['26 day EWA']
        P['Spread MACD'] = S['Spread MACD']
        #MACD = talib.MACD(S)
        #turn the spread into a signal
        
        P['SpreadSignal'] = np.where(S['Spread MACD'] > 0,1,0)
        #return prices, indicator, signal
        
        P = P.fillna(0)
        if back == True:
            backtest = self.metrics(P,'SpreadSignal')
            return P, backtest
        else:
            return P
    
    #bollinger bands 
    def BBands(self,method = "touch",back = False):
        #get coin prices in USD
       P = self.USDPrices
       P['MA'] = P['close'].rolling(20).mean()
       P['std'] = P['MA'].rolling(20).std()
       #one std bands
       x1 = talib.BBANDS(P['close'], nbdevup = 1,nbdevdn =1)
       #concat the bands into 1 df and change column names for each
       combine1 = pd.concat(x1,axis = 1)
       combine1.columns = ['upper1','middle1','lower1']
       
       #put it all together with prices
       P = pd.concat([P,combine1],axis = 1)
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
       
       P['bband%'] = (P['close'] - P['lower1'])/(P['upper1'] - P['lower1'])
       P = P.fillna(0)
       if back == True:
            backtest = self.metrics(P,'BBandsSignal')
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
        
        
        if back == True:
            backtest = self.metrics(P,'LagSignal')
            return P, backtest
        else:
            return P
    
    def trange(self,back = False):
        P = self.USDPrices
        Trange = talib.TRANGE(P['high'],P['low'],P['close'])
        P['true range'] = Trange.fillna(0)
        P['Trange signal'] = np.where(P['true range'] > P['true range'].shift(1),1,0)
        P['trange%'] = np.log(P['true range']/P['true range'].shift(1))

        if back == True:
            performance = self.metrics(P,'Trange signal')
            return P,performance
        elif back == False:
            return P
    #def 3blackcrows(self):
     #   P = self.USDPrices
      #  P['3 crows'] = talib.CDL3BLACKCROWS(P['open'], P['high'], P['low'], P['close'])
        
    #def takuri(self):
        #P = self.USDPrices
        #P['takuri'] = talib.CDLTAKURI(P['open'], P['high'], P['low'], P['close'])
        #P['tSignal'] = np.where(P['takuri'] > 0,1,np.where(P['takuri'].shift(1) >0,1,0))
    #def 3sstars(self):
    
    def metrics(self,strategy,name,fwd = False):
        if fwd == False:
            strategy['returns'] = np.log(strategy['close']/strategy['close'].shift(1)) 
            strategy['FwdReturns'] = strategy['returns'].shift(-1) 
        else:
            pass#- strategy['transaction']
        strategy['transaction'] = np.where(strategy[name] == 0,0,np.where(strategy[name].shift(1) < 1,np.where(strategy[name].shift(-1) <1,.001,.0005 ),np.where(strategy[name].shift(-1) == 1,0,.0005)))
        strategy['SFwdReturns'] = strategy['FwdReturns'] - strategy['transaction']
        df = strategy.fillna(0)
        r = df['FwdReturns'].values
        rS = df['SFwdReturns']
        s = df[name].values
        #s = np.where(s == 0,-1,1)
        rB = np.where(r> 0,1,0)
        accuracy = accuracy_score(rB, s)
        strat_r = (rS @ s) #- sum(s) * .0032
        ones = np.ones(len(r))
        coin_r = (r @ ones)
        alpha = strat_r - coin_r
        stdR = r.std()
        r = np.array_split(r,10)
        r = [x for x in r]
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
    Need to pass in the coin's prics in USD and coin's prices in another coin
    Just like strat, it needs the colummns open, low, high, and close. 
    
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
        df = self.USDPrices[['date','open','low','high','close']]
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
    """
    This class uses for complex mathematical techniques to prdict price movements
    Need to pass in the same dataframes as stratombine with the same column names
    
    """
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
    def modelFit(self,binary,mod,cv = False):
        scaler = StandardScaler()
        model = mod
        model2 = PCA(n_components = 4)
        #get the dataframe of all the indicators
        signal = super(complexStrat,self).signalCreateAll()
        signal2 = super(complexStrat,self).signalCreateAll()
        # calculate returns and forward returns
        signal2['returns'] = np.log(signal2['close']/signal2['close'].shift(1)) 
        signal2['FwdReturns'] = signal2['returns'].shift(-1)
        signal['returns'] = np.log(signal['close']/signal['close'].shift(1)) 
        signal['FwdReturns'] = signal['returns'].shift(-1) - .0005
        signal['binary'] = np.where(signal['FwdReturns'].fillna(0) > 0,1,0)
        sig = sum(signal['binary'].values)
        total = len(signal['binary'].values)
        decider = sig/total
        
        if decider > .5:
            up_majority = signal[signal['binary']==1]
            
            up_minority =signal[signal['binary']==0]
        else:
            up_majority = signal[signal['binary']==0]
            
            up_minority =signal[signal['binary']==1]
        
        #code to resample
        up_upsampled = resample(up_minority,replace = True,n_samples = len(up_majority),random_state = 123)
        signal = pd.concat([up_majority,up_upsampled])
        #get just the indicators for data running
        #if binary is true, only uses th signals, otherwise it is continuous
        if binary ==True:
            X = signal[['MACDsignal','SpreadSignal','BBandsSignal','LagSignal','Trange signal']]
        #continuous variables
        elif binary == False:
            
            X = signal[['MACD','bband%','Spread MACD','Lag Returns' ]].fillna(0)
            #transform independent variables
            X = X.astype(np.float32)
            
            X = scaler.fit_transform(X)
            
            X = model2.fit_transform(X)
            #get the y variable and make it binary
        y = np.where(signal['FwdReturns'].fillna(0) > 0,1,0)
    
        #split into training adn test data
        X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.50, random_state = 7 )
        #fit the model and predict
        model.fit(X_train,y_train)
        #predict and turn into a signal
        X2 = signal2[['MACD','bband%','Spread MACD','Lag Returns' ]].fillna(0)
        X2 = scaler.transform(X2)
        X2 = model2.transform(X2)
        predictions = model.predict(X2)
        signal2['signal'] = predictions
        y2 = np.where(signal2['FwdReturns'].fillna(0) > 0,1,0)
        
        if cv == True:
            n_splits = 10
            kfold = model_selection.KFold( n_splits , random_state = 7 )
            cv_results = model_selection.cross_val_score( model, X2, y2.ravel(), cv = kfold, scoring = 'accuracy' )
            return signal2, cv_results.mean() 
        if cv == False:
            return signal2
    def returns(self,signal,back,cv):
        if cv == True:
            signal,cross = signal
            
        #control whether metrics are returned
        if back == False:
            return signal
        elif back ==True:
            performance = super(complexStrat,self).metrics(signal, 'signal',fwd = True)
            if cv == True:
                performance["CV"] = cross
                return signal, performance
            else:
                return signal, performance
    def logistic(self,binary = False,back = False,cv = False):
        """
        returns dataframe of signals and indicators
        back: controls if it returns backtest metrics or not
        
        """
        signal = self.modelFit(binary,LogisticRegression(),cv = cv)
        
        if cv == True:
            signal,cross = signal
            
        #control whether metrics are returned
        if back == False:
            return signal
        elif back ==True:
            performance = super(complexStrat,self).metrics(signal, 'signal',fwd = True)
            if cv == True:
                performance["CV"] = cross
                return signal, performance
            else:
                return signal, performance
        
        
        
        
        
        
        
    def XGBoost(self,binary = False,back = False,cv = False):
        signal = self.modelFit(binary,XGBClassifier(max_depth = 3),cv = cv)
        #signal, cv = signal
        #get performance metrics
        if cv == True:
            signal, cross = signal
        #control whether metrics are returned
        if back == False:
            return signal
        elif back ==True:
            performance = super(complexStrat,self).metrics(signal, 'signal',fwd = True)
            if cv == True:
                performance["CV"] = cross
                return signal, performance
            else:
                return signal, performance
        
    def SVM(self, binary = False,back = False, cv = False):
        model = svm.SVC( kernel = 'rbf', C = 1, gamma = 10 )
        
        signal = self.modelFit(binary,model,cv )
        
        if cv == True:
            signal, cross = signal
        if back == False:
            return signal
        elif back ==True:
            performance = super(complexStrat,self).metrics(signal,'signal')
            if cv == True:
                performance["CV"] = cross
                return signal, performance
            else:
                return signal, performance
        
        
        
        
        
        
        
        