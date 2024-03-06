# Predicting Implicit Patterns and Optimizing Market Entry and Exit Decisions in Stock Prices using integrated Bayesian CNN-LSTM with Deep Q-Learning as a Meta-Labeller 

This is still an ongoing research project for MSc Financial Engineering.

**Submissions**

-  [![Final submission work]](https://github.com/adrikodebo/bayesian-CNN-LSTM-Q-learning/blob/main/final-bayesian-CNN-LSTM-Q-learning.ipynb)

 -  [![Draft 1]](https://github.com/adrikodebo/bayesian-CNN-LSTM-Q-learning/model-factory/model-1/draft1-bayesian-CNN-LSTM-Q-learning.ipynb)

**We are doing the following in this project**

- Train a Bayesian CNN-LSTM hybrid to predict the stock returns
- In the process estimate the uncertianty of the Bayesian CNN-LSTM prediction
- Use the Bayesian CNN--LSTM prediction and the uncertainty estimations as states to train Deep Q-learning agent (DQA).
- The purpose of training the DQA is to determine the size of bet based on Bayesian CNN-LSTM prediction and the uncertainty of the prediction of the Bayesian CNN-LSTM prediction

## Instruction

If you have an access clone the repository and create a virtual env and install the libraries from the requirements.txt file

**Note:** As of now the Deep Q-learning agents evaluation is still underway. I also, think there is still a room for fine-tuning the DQA and the bayesian 