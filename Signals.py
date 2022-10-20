# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 09:00:05 2021

@author: Lucas
"""

import numpy as np
import pandas as pd

import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scipy.stats.mstats import winsorize
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
    def __init__(self,USDPrices,coinPrices  ):
        self.USDPrices = USDPrices.rename(columns = str.lower)
        self.coinPrices =  coinPrices.rename(columns = str.lower)
    #MACD function
    def MACD(self, shortWindow = 12, longWindow = 26, back = False):
        
        #get the prices and only use close for ease of running
        P = self.USDPrices
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
       #turn ma and std to arrays
       upper = P['MA'].values + P['std'].values
       lower = P['MA'].values - P['std'].values
       #put it all together with prices
       #do this in an array
       #initialize the signal colum
       sig = np.zeros(len(P))
       close = P['close'].values
       #if the price touchs the top band, sell, and vice versa lower band
       # do this entire sequence in arrays
       if method == "touch":
           #for loop, change it row by row if the close interacts with the band
           for i in range(len(P)):
               if close[i] >= upper[i]:
                   sig[i] = 0
               elif close[i] <= lower[i]:
                   sig[i] = 1
       #if the price touches top band buy, and vice versa lower band
       elif method == "trend":
            #for loop, change it row by row if the close interacts with the band
            for i in range(len(P)):
               if close[i] >= upper[i]:
                   sig[i] = 1
               elif close[i] <= upper[i]:
                   sig[i] = 0
       
#       P['bband%'] = (P['close'] - P['lower1'])/(P['upper1'] - P['lower1'])
       B = (close - lower)/(upper - lower)
       P['BBandsSignal'] = sig
       P['bband%'] = B
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
        P = P.fillna(0)
        lag = P['Lag Returns'].values
        
        for i in range(len(P)):
            if method == "trend":
                sig = np.where(lag>0,1,0)
            elif method == "reversion":
                sig = np.where(lag<0,1,0)
        P['LagSignal'] = sig
        if back == True:
            backtest = self.metrics(P,'LagSignal')
            return P, backtest
        else:
            return P
    
    def trange(self,back = False):
        P = self.USDPrices
        P['trange'] = np.ones(len(P))
        for i in range(len(P)):
            if i == 0:
                pass
            else:
                P['trange'][i] = max((P['high'][i] - P['low'][i]),abs(P['high'][i] - P['close'][i-1]),abs(P['low'][i] - P['close'][i-1]))
        
        P['Atrange'] = P['trange'].rolling(20).mean()
        if back == True:
            performance = self.metrics(P,'Trange signal')
            return P,performance
        elif back == False:
            return P
    #def AVI(self,back = False):
     #   P = self.trange()
      #  P['DM+'] = P['high'] - P['high'].shift(1)
       # P['DM-'] = P['low'] - P['low'].shift(1)
        #P['smoothed +'] = 
        #P['smoothed -'] = 
    def OBV(self,back = False):
        P = self.prices
        P['obv'] = np.where(P['close'] > P['close'].shift(1),P['volume'] + P['volume'].shift(1),P['volume']-P['volume'].shift(1)) 
        P = P.fillna(0)
        P['obv signal'] = np.where(P['obv'] > P['obv'].shift(1),1,0)
        if back == True:
            performance = self.metrics(P,'obv signal')
            return P,performance
        elif back == False:
            return P
    def RSI(self,back = False):
        P = self.USDPrices
        P['r'] = np.log(P['close']/P['close'].shift(1))
        P['pos'] = np.ones(len(P))
        P['neg'] = np.zeros(len(P))
        for i in range(len(P)):
            if i < 14:
                 
                window = P['r'][:i+1]
        #print(window)
                
            else:
                window = P['r'][i-3:i+1]
        #print(window)
            x = window.values
            xPos = x[x>0]
            xNeg = x[x<0]

            P['pos'][i] = xPos.mean()
            P['neg'][i] = xNeg.mean()
        P['p/n'] = P['pos']/P['neg']
        P['RSI1'] = 100 - (100/(1 + (P['p/n'])))
        P['RSI'] = 100 - (100/(1 + (P['pos'].shift(1) * 13 + P['pos'])/(P['neg'].shift(1) * 13 + P['neg'])))
        P['RSI signal'] = np.where(P['RSI'].values > 70,1,0)
        if back == False:
            return P
        elif back == True:
            performance = self.metrics(P,'RSI signal')
            return P,performance
    #function to get all of the relavent metrics
    def metrics(self,strategy,name,fwd = False):
        """
        strategy: a dataframe in ascending order of price history
        name: the column name of the binary signal
        fwd: whether the forward returns are in the dataframe or not
        
        """
        #if not in the dataframe, calculates the forward returns
        if fwd == False:
            strategy['returns'] = np.log(strategy['close']/strategy['close'].shift(1)) 
            strategy['FwdReturns'] = strategy['returns'].shift(-1) 
        else:
            pass#- strategy['transaction']
        #subtracts transaction costs
        strategy['transaction'] = np.where(strategy[name] == 0,0,np.where(strategy[name].shift(1) < 1,np.where(strategy[name].shift(-1) <1,.001,.0007 ),np.where(strategy[name].shift(-1) == 1,0,.0007)))
        #subtracts transaction costs for the strategies forward returns
        strategy['SFwdReturns'] = strategy['FwdReturns'] - strategy['transaction']
        #fill the na with a 0
        df = strategy.fillna(0)
        #the coins forward returns
        r = df['FwdReturns'].values
        #strategy forward returns
        rS = df['SFwdReturns'].values
        #gets binary signals
        s = df[name].values
         
        #s = np.where(s == 0,-1,1)
        #turns the coins return into binary signal
        rB = np.where(r> 0,1,0)
        #finds accuracy score
        accuracy = accuracy_score(rB, s)
        precision = precision_score(rB, s)
        recall = recall_score(rB,s)
        f1 = f1_score(rB,s)
        confusion = confusion_matrix(rB,s)
        #strategy returns
        strat_r = (rS @ s) #- sum(s) * .0032
        #ones for coin returns
        ones = np.ones(len(r))
        #coin returns
        coin_r = (r @ ones)
        #alpha
        alpha = strat_r - coin_r
        #strandard deviation of returns
        stdR = r.std()
        #splits into periods to check for stability
        r = np.array_split(r,10)
        r = [x for x in r]
        ones = np.array_split(ones,10)
        rS = np.array_split(rS,10)
        s = np.array_split(s,10)
        #empy lists
        periodR = []
        periodC = []
        #calculates period by period returns for strategy and coin
        for i in range(len(r)):
            rP = (rS[i] @ s[i])
            rC = (r[i] @ ones[i])
            periodR.append(rP)
            periodC.append(rC)
        #turns into array
        R = np.array(periodR)
        C = np.array(periodC)
        #period by period alpha
        periodA = (R-C)
        #find the edge
        edge = (max(R) - abs(min(R)))/(max(R) + abs(min(R)))
        #turn all of the metrics into a dictionary and return it
        performance = {"Accuracy":accuracy,"precision":precision,"recall":recall,'f1':f1,"alpha":alpha,"PeriodByPeriodAlpha": periodA,"returns":strat_r,"returnsBtPeriod":R,'Edge': edge,"retrunStd": stdR,"confusion":confusion}
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
        trange = super(stratCombine, self).trange()[['Atrange']]
        #put it all into one dataframe
        RSI = super(stratCombine,self).RSI()[['RSI','RSI signal']]
        P = pd.concat([df,MACD,BBands,spread,SimpLagReturns,trange,RSI], axis = 1)
        
        return P
    #function to automatically filter the bad signals
    def signalFilter(self):
        pass
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
        signal['FwdReturns'] = signal['returns'].shift(-1) 
        signal['binary'] = np.where(signal['FwdReturns'].fillna(0) > 0,1,0)
        sig = sum(signal['binary'].values)
        total = len(signal['binary'].values)
        
        #determine if more up or more down days
        decider = sig/total
        #upsamples
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
            X = signal[['MACDsignal','SpreadSignal','BBandsSignal','LagSignal']]##,'RSI signal']]
        #continuous variables
        elif binary == False:
            
            X = signal[['MACD','bband%','Spread MACD','Lag Returns']]##,'RSI' ]].fillna(0)
            #transform independent variables
            X.replace([np.inf,-np.inf],0,inplace = True)
            
            X = X.fillna(0)
            
            X = X.astype(np.float32)
            X = X.values
            #X = X.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
            X = scaler.fit_transform(X)
            
            X = model2.fit_transform(X)
            #get the y variable and make it binary
        y = np.where(signal['FwdReturns'].fillna(0) > 0,1,0)
    
        #split into training adn test data
        X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.75, random_state = 7 )
        #fit the model and predict
        model.fit(X_train,y_train)
        #predict and turn into a signal
        if binary ==True:
            X2 = signal2[['MACDsignal','SpreadSignal','BBandsSignal','LagSignal']]##,'RSI signal']]
        else:
            X2 = signal2[['MACD','bband%','Spread MACD','Lag Returns' ]]##,'RSI']].fillna(0)
            X2 = scaler.transform(X2)
            X2 = model2.transform(X2)
        predictions = model.predict(X2)
        signal2['signal'] = predictions
        y2 = np.where(signal2['FwdReturns'].fillna(0) > 0,1,0)
        
        if cv == True:
            n_splits = 10
            kfold = model_selection.KFold( n_splits , shuffle = True,random_state = 7 )
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
        def analytics(self):
            pass
        def risk(self):
            pass
        
        