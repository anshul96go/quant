import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

# Importing libraries
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from plotly import tools
# import plotly.plotly as py
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error


class Model:

    def __init__(self):
        pass
    def create_target_variable(self):
        self.df['return'] = (self.df['f24'].shift(-78) - self.df['f24']) / self.df['f24']
        return df

    def prepare_features(self):
        """
        :param df: this is the data you want to use to prepare the features for your model
        :return: X, a matrix of features (can be a numpy array or a pandas dataframe, your choice!)
        """
        # todo: implement this function - you can use some of the features given to you or you can build a batch of
        #  your own based on the data that you are given.
        # *** PLEASE ENSURE THAT DO NOT INTRODUCE A LOOKAHEAD IN THIS MATRIX ***
        # *** Bonus points for coding a function that tests against lookahead in X ***\

        '''
        create_target_variable <done>
        handle mising value <not applicable in univariate case>
        '''
        # Target Variable
        self.df['return'] = (self.df['f24'].shift(-78) - self.df['f24']) / self.df['f24']
        # df = self.create_target_variable(df)

        return df

    def fit(self, path_to_train_csv, *args, **kwargs):
        # todo: read train csv
        # todo: do any operation you would like on it
        self.df = pd.read_csv(path_to_train_csv)

        # todo: prepare features for the model fit
        self.X = self.prepare_features(df)['return']
        

        # todo: fit your model here - use X (features matrix), y (the target variable) and any other information you
        #  want to use

        # this follows the scikit-learn pattern by returning self
        return self

    def predict(self, path_to_test_csv, *args, **kwargs):
        # todo: read test csv
        # todo: do any operation you would like on it

        # todo: prepare features for the model predict
        X = self.prepare_features(some_dataframe)

        # todo: calculate your model prediction (call it ypred) using X and any other information you want to use

        # this follows the scikit-learn pattern by returning ypred
        return ypred


df_orig = pd.read_csv('candidate.csv')
clf = Model()
df = df_orig.copy()
df = clf.create_target_variable(df)