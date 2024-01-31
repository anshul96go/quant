import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from statsmodels.tools.eval_measures import rmse
from sklearn.linear_model import Ridge

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

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


class Model:
    def get_target(self, df):
        df['return'] = (df['f24'].shift(-78) - df['f24']) / df['f24']
        df['return'] = df['return'].replace([np.inf, -np.inf], 0)
        return df[['return']]

    def prepare_features(self, df):
        """
        :param df: this is the data you want to use to prepare the features for your model
        :return: X, a matrix of features (can be a numpy array or a pandas dataframe, your choice!)
        """
        # todo: implement this function - you can use some of the features given to you or you can build a batch of
        #  your own based on the data that you are given.
        # *** PLEASE ENSURE THAT DO NOT INTRODUCE A LOOKAHEAD IN THIS MATRIX ***
        # *** Bonus points for coding a function that tests against lookahead in X ***

        ## 1. Data Transformation
        return_type_columns = ['f0', 'f1', 'f2', 'f3', 'f11', 'f12']
        price_type_columns = ['f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                              'f10', 'f13', 'f16', 'f17', 'f18', 'f19',
                              'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
                              'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39',
                              'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49',
                              'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59',
                              'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69',
                              'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76']
        integer_type_columns = ['f14', 'f15', 'f77', 'f78']

        for col in price_type_columns:
            df[col] = (df[col] - df[col].shift(78)) / df[col].shift(78)

        for col in integer_type_columns:
            df[col] = (df[col] - df[col].shift(78)) / df[col].shift(78)

        X = df

        # handle case of infinity
        X = X.replace([np.inf], 1)
        X = X.replace([-np.inf], -1)

        ## 2. Drop highly correlated variables
        X.drop(columns=['f5', 'f6', 'f7', 'f9', 'f8'], inplace=True)
        X.drop(columns=['f25', 'f26', 'f28', 'f29', 'f31', 'f32', 'f34', 'f35'], inplace=True)
        X.drop(columns=['f37', 'f38', 'f39', 'f42', 'f44', 'f45', 'f47', 'f48'], inplace=True)
        X.drop(columns=['f51', 'f52', 'f54', 'f55', 'f57', 'f58', 'f60', 'f61'], inplace=True)
        X.drop(columns=['f65', 'f64', 'f67', 'f63', 'f71', 'f70', 'f74', 'f73'], inplace=True)

        ## 3. Outlier Treatment
        window_size = 20 * 79  # 1 month (working days only)
        threshold = 2
        rolling_mean = X.rolling(window=window_size, min_periods=1).mean()
        rolling_std = X.rolling(window=window_size, min_periods=1).std()
        lower_bound = rolling_mean - threshold * rolling_std
        upper_bound = rolling_mean + threshold * rolling_std
        req_cols = list(X.columns)
        req_cols.remove('return')
        print("req_cols without 'return' ", req_cols)
        for column in req_cols:
            X[column] = X[column].clip(lower=lower_bound[column], upper=upper_bound[column], axis=0)

        # include lag column
        #         column_name = 'return'
        for column_name in X.columns:
            for i in range(1, self.lag + 1):
                lagged_column_name = f'{column_name}_lag_{i}'
                X[lagged_column_name] = X[column_name].shift(i * 78)

        # delete return columns
        del X['return']

        return X

    #     def create_lagged_columns(df, column_name, lag_order):
    #         for i in range(1, lag_order + 1):
    #             lagged_column_name = f'{column_name}_lag_{i}'
    #             df[lagged_column_name] = df[column_name].shift(i)

    #     # Example: create 2 lagged columns for 'return'
    #     create_lagged_columns(df, 'return', 2)

    def read_data(self, path_to_data):
        data = pd.read_csv(path_to_data, index_col='time', parse_dates=['time'])
        data.index = pd.to_datetime(data.index, format='%d-%m-%Y %H:%M')
        data.sort_index(inplace=True)
        data = data.fillna(method='ffill')
        return data

    def fit(self, path_to_train_csv, *args, **kwargs):
        """
        ### AG:  TASKS
        ## Model Selection:
            Linear: Base Model
            Ridge : Handles Multicollinearity
            RandomForest (large number of uncorrelated features, fail if the potential y values lie outside)
            Time Series
        ## Train-Test Split to get the optimal model
        ## Train complete model
        ## Store the optimal model
        """

        # get the values from kwargs
        self.alphas = kwargs['alphas']
        self.lag = kwargs['lag']
        self.l1_ratio = kwargs['l1_ratio']

        # todo: read train csv
        # todo: do any operation you would like on it
        self.train = self.read_data(path_to_train_csv)

        # todo: prepare features for the model fit
        self.y = self.get_target(self.train)
        self.X = self.prepare_features(self.train)

        ## AG: Drop missing values
        combined = pd.concat([self.X, self.y], axis=1)
        combined_clean = combined.dropna()
        self.X_clean = combined_clean.drop(columns='return')
        self.y_clean = combined_clean['return']

        # Sequential split into train, cv, and test sets
        total_length = len(self.X_clean)
        train_size = int(total_length * 0.8)  # 60% of data for training
        #         test_size = int(total_length * 0.2)     # 20% of data for cross-validation

        X_train, y_train = self.X_clean[:train_size], self.y_clean[:train_size]
        X_test, y_test = self.X_clean[train_size:], self.y_clean[train_size:]

        # Range of hyperparameters to test for Lasso, Ridge, and Elastic Net
        best_alpha = None
        lowest_cv_rmse = float('inf')
        best_model = None
        lowest_rmse = float('inf')
        best_model_name = None

        res_list = []

        for model_type in args:
            if model_type == 'ols':
                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = pd.Series(np.nan, index=X_test.index)
                non_nan_rows = ~X_test.isnull().any(axis=1)
                y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                #                 print(y_pred)
                #                 print(y_test)
                test_rmse = get_rmse(y_test, y_pred)
                res_list.append({'model': model_type, 'lag': self.lag, 'alpha': None, 'test_rmse': test_rmse})

                if test_rmse < lowest_rmse:
                    lowest_rmse = test_rmse
                    best_model = model
                    best_model_name = model_type

            else:
                total_length = len(X_train)
                train_size = int(total_length * 0.8)
                X_train_fold, y_train_fold = X_train[:train_size], y_train[:train_size]
                X_cv_fold, y_cv_fold = X_train[train_size:], y_train[train_size:]

                # test for lasso model
                if model_type == 'lasso':
                    lowest_cv_rmse = float('inf')
                    for alpha in self.alphas:
                        model = Lasso(alpha=alpha)
                        model.fit(X_train_fold, y_train_fold)
                        y_cv_pred = pd.Series(np.nan, index=X_cv_fold.index)
                        non_nan_rows = ~X_cv_fold.isnull().any(axis=1)
                        y_cv_pred[non_nan_rows] = model.predict(X_cv_fold[non_nan_rows])
                        cv_rmse = get_rmse(y_cv_fold, y_cv_pred)
                        #                         print("model: {}, alpha: {}, cv_rmse: {}".format(model_type,alpha,cv_rmse))
                        if cv_rmse < lowest_cv_rmse:
                            best_alpha = alpha
                            lowest_cv_rmse = cv_rmse

                    # get the test_rmse for the model
                    model = Lasso(alpha=best_alpha)
                    model.fit(X_train, y_train)
                    y_pred = pd.Series(np.nan, index=X_test.index)
                    non_nan_rows = ~X_test.isnull().any(axis=1)
                    y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                    test_rmse = get_rmse(y_test, y_pred)

                    res_list.append({'model': model_type, 'lag': self.lag, 'alpha': best_alpha, 'test_rmse': test_rmse})
                    if test_rmse < lowest_rmse:
                        lowest_rmse = test_rmse
                        best_model = model
                        best_model_name = model_type

                # test for ridge model
                if model_type == 'ridge':
                    lowest_cv_rmse = float('inf')
                    for alpha in self.alphas:
                        print('alpha: ',alpha)
                        model = Ridge(alpha=alpha)
                        model.fit(X_train_fold, y_train_fold)
                        y_cv_pred = pd.Series(np.nan, index=X_cv_fold.index)
                        non_nan_rows = ~X_cv_fold.isnull().any(axis=1)
                        y_cv_pred[non_nan_rows] = model.predict(X_cv_fold[non_nan_rows])
                        cv_rmse = get_rmse(y_cv_fold, y_cv_pred)
                        print("model: {}, alpha: {}, cv_rmse: {}".format(model_type,alpha,cv_rmse))
                        if cv_rmse < lowest_cv_rmse:
                            best_alpha = alpha
                            lowest_cv_rmse = cv_rmse

                    # get the test_rmse for the model
                    model = Ridge(alpha=best_alpha)
                    model.fit(X_train, y_train)
                    y_pred = pd.Series(np.nan, index=X_test.index)
                    non_nan_rows = ~X_test.isnull().any(axis=1)
                    y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                    test_rmse = get_rmse(y_test, y_pred)

                    res_list.append({'model': model_type, 'lag': self.lag, 'alpha': best_alpha, 'test_rmse': test_rmse})
                    if test_rmse < lowest_rmse:
                        lowest_rmse = test_rmse
                        best_model = model
                        best_model_name = model_type

                # test for ridge model
                if model_type == 'elastic_net':
                    lowest_cv_rmse = float('inf')
                    best_l1_ratio = 0
                    for alpha in self.alphas:
                        for l1_ratio in self.l1_ratio:
                            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                            model.fit(X_train_fold, y_train_fold)
                            y_cv_pred = pd.Series(np.nan, index=X_cv_fold.index)
                            non_nan_rows = ~X_cv_fold.isnull().any(axis=1)
                            y_cv_pred[non_nan_rows] = model.predict(X_cv_fold[non_nan_rows])
                            cv_rmse = get_rmse(y_cv_fold, y_cv_pred)
                            #                             print("model: {}, alpha: {}, l1_ratio: {},cv_rmse: {}".format(model_type,alpha,l1_ratio,cv_rmse))
                            if cv_rmse < lowest_cv_rmse:
                                best_alpha = alpha
                                best_l1_ratio = l1_ratio
                                lowest_cv_rmse = cv_rmse

                        # get the test_rmse for the model
                        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
                        model.fit(X_train, y_train)
                        y_pred = pd.Series(np.nan, index=X_test.index)
                        non_nan_rows = ~X_test.isnull().any(axis=1)
                        y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                        test_rmse = get_rmse(y_test, y_pred)

                        res_list.append({'model': model_type, 'lag': self.lag, 'alpha': (best_alpha, best_l1_ratio),
                                         'test_rmse': test_rmse})
                        if test_rmse < lowest_rmse:
                            lowest_rmse = test_rmse
                            best_model = model
                            best_model_name = model_type

        self.model = best_model
        self.model_name = best_model_name
        self.res_list = res_list
        return self

    def predict(self, path_to_test_csv, *args, **kwargs):
        # todo: read test csv
        # todo: do any operation you would like on it
        self.test = pd.read_csv(path_to_test_csv, index_col='time', parse_dates=['time'])
        self.test.index = pd.to_datetime(self.test.index, format='%d-%m-%Y %H:%M')
        self.test.sort_index(inplace=True)
        self.test = self.test.fillna(method='ffill')

        # todo: prepare features for the model predict
        self.y_test = self.get_target(self.test)
        self.X_test = self.prepare_features(self.test)

        # todo: calculate your model prediction (call it ypred) using X and any other information you want to use
        ypred = pd.Series(np.nan, index=self.X_test.index)
        non_nan_rows = ~self.X_test.isnull().any(axis=1)
        ypred[non_nan_rows] = self.model.predict(self.X_test[non_nan_rows])

        # this follows the scikit-learn pattern by returning ypred
        return ypred


def get_rmse(ypred, ytest):
    combined = pd.concat([ypred, ytest], axis=1)
    combined_clean = combined.dropna()

    rmse_ = rmse(combined_clean[0], combined_clean['return'])
    return rmse_

if __name__ == '__main__':
    train_csv_path = 'train.csv'
    test_csv_path = 'test.csv'

    fit_args = ['ols','lasso','ridge','elastic_net']   # todo: populate this as you see fit
    fit_kwargs = {'alphas':np.linspace(0,1,11),'lag':2,'l1_ratio':np.linspace(0,1,11)}  # todo: populate this as you see fit
    clf = Model()
    clf.fit(train_csv_path, *fit_args, **fit_kwargs)

    predict_args = []  # todo: populate this as you see fit
    predict_kwargs = {}  # todo: populate this as you see fit
    ypred = clf.predict(test_csv_path, *predict_args, **predict_kwargs)
    print(get_rmse(ypred, clf.y_test))
