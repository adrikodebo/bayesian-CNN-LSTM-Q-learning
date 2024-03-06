import datetime
import math

import numpy as np
import pandas_datareader.data as web
import seaborn as sns
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox,probplot,shapiro
# from sklearn_fracdiff import FracDiff
from sklearn.model_selection import train_test_split
from pyts.image import GramianAngularField
from keras.models import Sequential
from keras.layers import Conv1D,Bidirectional, MaxPooling1D, Flatten, Dense,Reshape,LSTM,Dropout,TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from keras.optimizers import Adam
from keras import backend as keras_b
import tensorflow as tf
from keras.models import load_model
import random
from collections import deque
import pickle
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (16, 9)

class BayesianCNNLSTM(object):
    """docstring for BayesianCNNLSTM"""
    def __init__(self, **kwargs):
        
        pass

    def model(self,):
        # For creating model and training
        model_bay_cnn_lstm = Sequential()

        # Creating the Neural Network model here...
        # CNN layers
        model_bay_cnn_lstm.add(
            TimeDistributed(
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 1,100, 1)
                      )
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                Dropout(0.25)
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                MaxPooling1D(2)
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                Conv1D(128, kernel_size=3, activation='relu')
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                MaxPooling1D(2)
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                Conv1D(64, kernel_size=3, activation='relu')
            )
        )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                MaxPooling1D(2)
                )
            )
        model_bay_cnn_lstm.add(
            TimeDistributed(
                Flatten()
                )
            )

        # LSTM layers
        model_bay_cnn_lstm.add(
            Bidirectional(
                LSTM(100, return_sequences=True)
                )
            )
        model_bay_cnn_lstm.add(
            Bidirectional(
                LSTM(100, return_sequences=False)
                )
            )
        model_bay_cnn_lstm.add(Dropout(0.5))

        #Final layers
        model_bay_cnn_lstm.add(Dense(1, activation='linear'))
        model_bay_cnn_lstm.compile(
            optimizer='adam', 
            loss='mse', 
            metrics=['mse', 'mae']
            )
        return model_bay_cnn_lstm
                