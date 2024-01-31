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

    def prepare_features(self, df, lag):
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

        if self.feature_engineering == True:
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
        if self.outlier_treatment == True:
            window_size = 20 * 79  # 1 month (working days only)
            threshold = 2
            rolling_mean = X.rolling(window=window_size, min_periods=1).mean()
            rolling_std = X.rolling(window=window_size, min_periods=1).std()
            lower_bound = rolling_mean - threshold * rolling_std
            upper_bound = rolling_mean + threshold * rolling_std
            req_cols = list(X.columns)
            req_cols.remove('return')
            #         print("req_cols without 'return' ", req_cols)
            for column in req_cols:
                X[column] = X[column].clip(lower=lower_bound[column], upper=upper_bound[column], axis=0)

        # include lag columns
        if self.X_lag == True:
            for column_name in X.columns:
                for i in range(1, lag + 1):
                    lagged_column_name = f'{column_name}_lag_{i}'
                    X[lagged_column_name] = X[column_name].shift(i * 78)
        else:
            column_name = 'return'
            for i in range(1, lag + 1):
                lagged_column_name = f'{column_name}_lag_{i}'
                X[lagged_column_name] = X[column_name].shift(i * 78)

        # delete return columns
        del X['return']

        return X

    def read_data(self, path_to_data):
        data = pd.read_csv(path_to_data, index_col='time', parse_dates=['time'])
        data.index = pd.to_datetime(data.index, format='%d-%m-%Y %H:%M')
        data.sort_index(inplace=True)
        data = data.fillna(method='ffill')
        return data

    def fit(self, path_to_train_csv, *args, **kwargs):

        # get the values from kwargs
        self.alphas = kwargs['alphas']
        self.lags = kwargs['lags']
        self.l1_ratio = kwargs['l1_ratio']
        self.X_lag = kwargs['X_lag']
        self.outlier_treatment = kwargs['outlier_treatment']
        self.feature_engineering = kwargs['feature_engineering']

        # Range of hyperparameters to test for Lasso, Ridge, and Elastic Net
        best_alpha = None
        best_l1_ratio = None
        best_model = None
        best_model_name = None
        best_lag = None
        lowest_rmse = float('inf')
        res_list = []

        # iterate across lags
        for lag in self.lags:
            self.train = self.read_data(path_to_train_csv)
            self.y = self.get_target(self.train)

            self.X = self.prepare_features(self.train, lag)

            ## AG: Drop missing values
            combined = pd.concat([self.X, self.y], axis=1)
            combined_clean = combined.dropna()
            self.X_clean = combined_clean.drop(columns='return')
            self.y_clean = combined_clean['return']

            total_length = len(self.X_clean)
            train_size = int(total_length * 0.8)
            X_train, y_train = self.X_clean[:train_size], self.y_clean[:train_size]
            X_test, y_test = self.X_clean[train_size:], self.y_clean[train_size:]

            for model_type in args:
                if model_type == 'ols':
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    #                     print("model.coef_: ",model.coef_)
                    y_pred = pd.Series(np.nan, index=X_test.index)
                    non_nan_rows = ~X_test.isnull().any(axis=1)
                    y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                    test_rmse = get_rmse(y_test, y_pred)
                    res_list.append(
                        {'model': model_type, 'lag': lag, 'alpha': None, 'l1_ratio': None, 'test_rmse': test_rmse})
                    if test_rmse < lowest_rmse:
                        best_alpha = None
                        best_l1_ratio = None
                        best_model = model
                        best_model_name = model_type
                        best_lag = lag
                        lowest_rmse = test_rmse

                else:
                    total_length = len(X_train)
                    train_size = int(total_length * 0.8)
                    X_train_fold, y_train_fold = X_train[:train_size], y_train[:train_size]
                    X_cv_fold, y_cv_fold = X_train[train_size:], y_train[train_size:]

                    # test for lasso model
                    if model_type == 'lasso':
                        lowest_cv_rmse = float('inf')
                        best_model_alpha = None
                        for alpha in self.alphas:
                            #                             print(alpha)
                            #                             print(X_train_fold,y_train_fold)
                            model = Lasso(alpha=alpha)
                            model.fit(X_train_fold, y_train_fold)
                            y_cv_pred = pd.Series(np.nan, index=X_cv_fold.index)
                            non_nan_rows = ~X_cv_fold.isnull().any(axis=1)
                            y_cv_pred[non_nan_rows] = model.predict(X_cv_fold[non_nan_rows])
                            cv_rmse = get_rmse(y_cv_fold, y_cv_pred)
                            if cv_rmse < lowest_cv_rmse:
                                best_model_alpha = alpha
                                lowest_cv_rmse = cv_rmse

                        # get the test_rmse for the model
                        model = Lasso(alpha=best_model_alpha)
                        model.fit(X_train, y_train)
                        y_pred = pd.Series(np.nan, index=X_test.index)
                        non_nan_rows = ~X_test.isnull().any(axis=1)
                        y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                        test_rmse = get_rmse(y_test, y_pred)

                        res_list.append({'model': model_type, 'lag': lag, 'alpha': best_model_alpha, 'l1_ratio': None,
                                         'test_rmse': test_rmse})
                        if test_rmse < lowest_rmse:
                            best_alpha = best_model_alpha
                            best_l1_ratio = None
                            best_model = model
                            best_model_name = model_type
                            best_lag = lag
                            lowest_rmse = test_rmse

                    # test for ridge model
                    if model_type == 'ridge':
                        lowest_cv_rmse = float('inf')
                        best_model_alpha = None
                        for alpha in self.alphas:
                            model = Ridge(alpha=alpha)
                            model.fit(X_train_fold, y_train_fold)
                            y_cv_pred = pd.Series(np.nan, index=X_cv_fold.index)
                            non_nan_rows = ~X_cv_fold.isnull().any(axis=1)
                            y_cv_pred[non_nan_rows] = model.predict(X_cv_fold[non_nan_rows])
                            cv_rmse = get_rmse(y_cv_fold, y_cv_pred)
                            #                         print("model: {}, alpha: {}, cv_rmse: {}".format(model_type,alpha,cv_rmse))
                            if cv_rmse < lowest_cv_rmse:
                                best_model_alpha = alpha
                                lowest_cv_rmse = cv_rmse

                        # get the test_rmse for the model
                        model = Ridge(alpha=best_model_alpha)
                        model.fit(X_train, y_train)
                        y_pred = pd.Series(np.nan, index=X_test.index)
                        non_nan_rows = ~X_test.isnull().any(axis=1)
                        y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                        test_rmse = get_rmse(y_test, y_pred)

                        res_list.append({'model': model_type, 'lag': lag, 'alpha': best_model_alpha, 'l1_ratio': None,
                                         'test_rmse': test_rmse})
                        if test_rmse < lowest_rmse:
                            best_alpha = best_model_alpha
                            best_l1_ratio = None
                            best_model = model
                            best_model_name = model_type
                            best_lag = lag
                            lowest_rmse = test_rmse

                    # test for ridge model
                    if model_type == 'elastic_net':
                        lowest_cv_rmse = float('inf')
                        best_model_l1_ratio = 0
                        best_model_alpha = None
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
                                    best_model_alpha = alpha
                                    best_model_l1_ratio = l1_ratio
                                    lowest_cv_rmse = cv_rmse

                        # get the test_rmse for the model
                        model = ElasticNet(alpha=best_model_alpha, l1_ratio=best_model_l1_ratio)
                        model.fit(X_train, y_train)
                        y_pred = pd.Series(np.nan, index=X_test.index)
                        non_nan_rows = ~X_test.isnull().any(axis=1)
                        y_pred[non_nan_rows] = model.predict(X_test[non_nan_rows])
                        test_rmse = get_rmse(y_test, y_pred)

                        res_list.append({'model': model_type, 'lag': lag, 'alpha': best_model_alpha,
                                         'l1_ratio': best_model_l1_ratio, 'test_rmse': test_rmse})
                        if test_rmse < lowest_rmse:
                            best_alpha = best_model_alpha
                            best_l1_ratio = best_model_l1_ratio
                            best_model = model
                            best_model_name = model_type
                            best_lag = lag
                            lowest_rmse = test_rmse

        self.model = best_model
        self.model_name = best_model_name
        self.res_list = res_list
        self.lag = best_lag
        self.alpha = best_alpha
        self.l1_ratio = best_l1_ratio
        self.rmse = lowest_rmse

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
        #         print("self.lag: ",self.lag)
        self.X_test = self.prepare_features(self.test, self.lag)

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
    # filename = 'no_xlag_no_out_no_fe'
    fit_args = ['ols','lasso','ridge','elastic_net']  # these 4 models can be used
    fit_kwargs = {'alphas':np.linspace(0,1,11),'lags':[0,1,3,5],'l1_ratio':np.linspace(0,1,11),'X_lag':True,
                  'outlier_treatment':True, 'feature_engineering':True}  # change parameters here

    train_csv_path = 'train.csv' # change the dataset path here
    test_csv_path = 'test.csv'   # change the dataset path here

    # fit the model
    clf = Model()
    clf.fit(train_csv_path, *fit_args, **fit_kwargs)

    # make predictions
    predict_args = []  # todo: populate this as you see fit
    predict_kwargs = {}  # todo: populate this as you see fit
    ypred = clf.predict(test_csv_path, *predict_args, **predict_kwargs)

    # calculates rmse on test data
    print(get_rmse(ypred, clf.y_test))