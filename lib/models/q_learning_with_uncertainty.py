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


# Function to get states from your Bayesian CNN-LSTM model and incorporate uncertainty
def get_states(y_pred,uncertainty):
    # create an extended state that incorporated uncertainty estimation
    # and the predictions
    # We limit the state size to 100*2
    # This is necessary to maintain consistency
    extended_state = np.concatenate((y_pred[:100,:], uncertainty[:100,:]), axis=1)
    return extended_state

# The DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _huber_loss(self,y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = keras_b.abs(error) <= clip_delta

        squared_loss = 0.5 * keras_b.square(error)
        quadratic_loss = (
                    0.5 * keras_b.square(clip_delta) +
                    clip_delta * (keras_b.abs(error) - clip_delta)
                )

        return keras_b.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(state_size, 2)))  # Flatten layer to reshape input
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(
            loss=self._huber_loss, 
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        # model = Sequential()
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # print(state.shape)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,current_holding=0):
        if np.random.rand() <= self.epsilon:
            # Randomly select action from the action space: short, do nothing, long
            return np.random.uniform(-1, 1)
        else:
            # Use the Q-network to select action based on state
            return np.argmax(self.model.predict(state,verbose=0)[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # print(next_state.shape)
            if not done:
                # print(self.model.predict(next_state))
                target = reward + self.gamma * np.amax(self.model.predict(next_state,verbose=0)[0])
            # print(state.shape)
            target_f = self.model.predict(state,verbose=0)
            # print(action)
            target_f[0][0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# The Market environment
class MarketEnvironment:
    def __init__(self, max_episode_length=100, min_available_capital=10, max_trades=None, profit_target=None, stop_loss=-5000):
        self.state_size = state_size  # The actual size of your state
        self.action_size = 1  # Example: Buy, Sell, Hold
        self.initial_capital = 10000  # Initial available capital
        self.transaction_cost = 0.01  # Transaction charge (1%)
        self.current_capital = self.initial_capital
        self.max_episode_length = max_episode_length
        self.min_available_capital = min_available_capital
        self.max_trades = max_trades
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.num_trades = 0
        self.total_reward = 0
        self.uncertainty_penalty=0.5

    def reset(self):
        # Reset logic
        self.current_capital = self.initial_capital
        self.num_trades = 0
        self.total_reward = 0
        # print(self.state_size)
        return np.random.rand(self.state_size)

    def step(self, action,time_state):
        # Extract market returns and uncertainty from the state
        market_returns = time_state[0]  # First column contains predicted market returns
        uncertainty = time_state[1]     # Second column contains uncertainty estimations
        # Simulated step function, returns next_state, reward, done
        next_state = np.random.rand(self.state_size,2)

        # Calculate bet size based on the selected action
        bet_size = abs(action) * self.current_capital

        # Calculate transaction cost based on the bet size
        total_transaction_cost=self.transaction_cost * bet_size
        # Subtract the transaction cost
        self.current_capital -= total_transaction_cost
        # Calculate the return based on the market return and the direction of the trade
        # For a long position, return is market return * bet size
        # For not taking any position, return is 0
        if action > 0:
            return_direction = 1
        elif action<0:
            return_direction = -1
        else:
            return_direction = 0
        return_amount = return_direction * market_returns * bet_size
        yield_size=return_amount+bet_size
        if action > 0:
            # Adjust available capital based on bet size
            self.current_capital += return_amount
    
        # If action == 0, it means the agent wants to close the position
        elif action == 0:
            # Adjust available capital based on the current holding
            self.current_capital += return_amount
        # Calculate reward based on market return, transaction charges, and uncertainty
        reward = return_amount
        reward -= self.uncertainty_penalty * uncertainty
        # Increment total reward
        self.total_reward += reward

        # Increment number of trades
        self.num_trades += 1
         # Check termination conditions
        done = False
        if self.num_trades >= self.max_episode_length:
            done = True
        elif self.current_capital <= self.min_available_capital:
            done = True
        elif self.max_trades is not None and self.num_trades >= self.max_trades:
            done = True
        elif self.profit_target is not None and self.total_reward >= self.profit_target:
            done = True
        elif self.stop_loss is not None and self.total_reward <= self.stop_loss:
            done = True
        return next_state, reward, done
