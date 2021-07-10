from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np


from env.gymWrapper import create_btc_env
from drqn_agent import DQNAgent
import os


import pandas as pd
import numpy as np
import math
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import EMAIndicator, ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

import tensorflow as tf
from tensorflow import keras

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(
    tf.compat.v1.Session(config=config))

class DADOS():    

    def __init__(self):
        
        names = ['Data','open','high','low','close', 'volume','open_win','high_win','low_win','close_win', 'volume_win']
        raw_df= pd.read_csv('WDO_WIN_15min.csv' , header = 0, names = names,  sep = ';', index_col=False)
        print(raw_df)        
        self.df = raw_df  
        #self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        indicator_bb = BollingerBands(close=self.df["close"], window=20, window_dev=2)
        vwap = VolumeWeightedAveragePrice(high=self.df["high"], low=self.df["low"], close=self.df["close"], volume=self.df["volume"] )
        ema9 = EMAIndicator(close=self.df["close"], window=8)
        ema21 = EMAIndicator(close=self.df["close"], window=17 )
        ema200 = EMAIndicator(close=self.df["close"], window=72 )
        ema9_win = EMAIndicator(close=self.df["close_win"], window=9 )
        ema21_win = EMAIndicator(close=self.df["close_win"], window=21 )
        #ema200_win = EMAIndicator(close=self.df["close_win"], window=9, fillna=True )
        rsi = RSIIndicator(close=self.df["close"])
        adx = ADXIndicator(high=self.df["high"], low=self.df["low"], close=self.df["close"], window = 8)
        volume_21 = SMAIndicator(close=self.df["volume"], window=21 )
        #estocastico = StochasticOscillator(high=self.df["high"], low=self.df["low"], close=self.df["close"], fillna=True)
        self.df['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.df['bb_bbh'] = indicator_bb.bollinger_hband()
        self.df['bb_bbl'] = indicator_bb.bollinger_lband()
        self.df['vwap'] = vwap.volume_weighted_average_price()
        self.df['ema9'] = ema9.ema_indicator()
        self.df['ema21'] = ema21.ema_indicator() 
        self.df['ema200'] = ema200.ema_indicator()
        self.df['ema9_win'] = ema9_win.ema_indicator()
        self.df['ema21_win'] = ema21_win.ema_indicator()
        
        self.df['rsi'] = rsi.rsi()
        self.df['adx'] = adx.adx() 
        self.df['-di'] = adx.adx_neg()
        self.df['+di'] = adx.adx_pos()
        self.df['volume_21'] = volume_21.sma_indicator()
        self.df = self.df.dropna()
        #self.df['estocastico']  = estocastico.stoch()
        #self.df['estocastico_signal'] = estocastico.stoch_signal()
        print(self.df)
        #self.df = self.df[['open','high','low','close', 'bb_bbh', 'bb_bbl', 'ema9', 'ema21', 'ema200','rsi','adx','-di', '+di', 'estocastico', 'estocastico_signal']].values

    def separa_dados_train(self, index: int, gap_train: int, gap_test: int, time:int):
        #tem 36 candles 1 dia  - 1253 DIAS NA TABELA TODA
        train = self.df[index-time : index+gap_train+time] #0+720+30+30=780 750ticks
        test = self.df[index+gap_train-time : index+gap_train+gap_test+time]
        return train, test, index + gap_test

        
    
def main():
    index = 30
    gap_train = 36*5*4*2 #36*5*4 #1mes #72*15 #2 dias 72 #15 MINUTOS
    gap_test = 36*5*7 #1 dia 36 #15 MINUTOS
    #gap_train = 108*5*4 #1mes #5 MINUTOS
    #gap_test = 108*5*7 #1 dia 36 #5 MINUTOS
    TIMESTEP = 30 #30
    data = DADOS()
    filename = ''
    
    for N in range(1):
                
        train, test, index_pos_new = data.separa_dados_train(index=index, gap_train = gap_train, gap_test = gap_test, time = TIMESTEP)
        print(train)
        print(test)
        index = index_pos_new 
        train_env = create_btc_env(window_size=TIMESTEP, path=train,train=True)
        if len(filename)==0:
            agent = DQNAgent(train_env, window_size_from_env = TIMESTEP)
        else:
            agent = DQNAgent(train_env, window_size_from_env = TIMESTEP, policy_network = tf.keras.models.load_model("agents/"+filename))
        
        #agent = DQNAgent(train_env, window_size_from_env = TIMESTEP)    
        filename = agent.train(n_steps=5000, n_episodes=1750, save_path="agents/")
        print(filename)
        print(len(filename))
        test_env = create_btc_env(window_size=TIMESTEP, path=test, train=True)
        agent_test = DQNAgent(test_env, window_size_from_env = TIMESTEP, policy_network = tf.keras.models.load_model("agents/"+filename))
        memory = agent_test.test(gap_test, n_episodes=3)
        #print(memory)
        print(index_pos_new)
        print(test)
        train_env = None
        test_env = None
        agent = None
        agent_test = None



    
if __name__ == '__main__':
    main()


