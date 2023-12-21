import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import gpflow
import scipy
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score, auc, r2_score
from typing import Tuple, Dict
from tqdm import trange


class Ornstein_Uhlenbeck(gpflow.kernels.IsotropicStationary):
    def K_r2(self, r2):
        return self.variance * tf.exp(-tf.sqrt(r2))


def line_fit(df: pd.DataFrame, target_col: str, window_size: list) -> Tuple:
    '''
    Simple regression with Y-values at target_col ran for each N consecutive rows of the data frame.
    
    Inputs:
    -------
        - df (pd.DataFrame): data frame with all needed data (Y-values).
        - target_col (str): name of the target column (i.e., column with Y-values).
        - window_size (int): size of the window for linear regression.
    Outputs:
    -------
        - coefs (list): list with values of linear regression coefficients for each window.
        - r2s (list): list with values of linear regression coefficients for each window.
    '''
    # fillna for reg
    df[target_col] = df[target_col].fillna(0)
    
    coefs = [np.nan] * window_size
    r2s = [np.nan] * window_size
    
    X = np.arange(window_size).reshape(-1, 1)
    for i in trange(window_size, len(df)):
        # Y prep
        y = df[target_col][i - window_size: i]
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        pred = reg.predict(X)
        r2 = r2_score(y, pred)
        coefs.append(reg.coef_[0])
        r2s.append(r2)
    return coefs, r2s


def get_inv_cov(samp_size: int) -> Dict:
    '''
    Get inversed square-root covariance matrix dictionary for different lengthscales for
    Matern and Ornstein-Uhlenbeck kernels.
    
    Inputs:
    -------
        - samp_size (int): sample size for the covariance matrix generation
    
    Outputs:
    -------
        - inv_cov_dct (dict): dictionary with inversed square-root covariance matrices 
                              for Matern and OU kernels
    '''
    
    inv_cov_dct = {}

    Xplot = np.arange(0, samp_size, 1).astype(float)[:, None]
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))

    kernels = {'Matern': gpflow.kernels.Matern32, 'Ornstein_Uhlenbeck': Ornstein_Uhlenbeck}

    for k_type in kernels.keys():
        inv_cov_dct_k = {}
        for ls in trange(5, 110, 5):
            k = kernels[k_type](lengthscales=ls, variance=1)
            model = gpflow.models.GPR((X, Y), kernel=k)
            _, cov = model.predict_f(Xplot, full_cov=True)

            cov = cov[0, :, :].numpy()
            inv_cov_dct_k[ls] = scipy.linalg.fractional_matrix_power(cov, -0.5)
        inv_cov_dct[k_type] = inv_cov_dct_k
    
    return inv_cov_dct


def plot_roc_auc(features_data: pd.DataFrame, column_name: str) -> None:
    '''
    Plot ROC curve for the given feature.
    
    Inputs:
    -------
    - features_data (pd.DataFrame): features data frame
    - column_name (str): name of the selected feature 
    Outputs:
    -------
        None
    '''
    
    dd = features_data[features_data['target'].notna()]
    dd['target'] = dd['target'].replace(-1, 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    dd_train = dd[:2000].sort_values(by=column_name).reset_index(drop=True)
    fprs, tprs, _ = sklearn.metrics.roc_curve(dd_train['target'], dd_train[column_name])
    plt.plot(1 - tprs, 1 - fprs, color='black')
    plt.plot([0, 1], [0, 1], 
             color='gray', 
             linestyle='--')

    plt.title('Train data \n AUC: ' + str(round(100 * auc(1 - tprs, 1 - fprs), 1)))

    plt.subplot(1, 2, 2)
    dd_test = dd[2000:].sort_values(by=column_name).reset_index(drop=True)
    fprs, tprs, _ = sklearn.metrics.roc_curve(dd_test['target'], dd_test[column_name])
    plt.plot(1 - tprs, 1 - fprs, color='black')
    plt.plot([0, 1], [0, 1], 
             color='gray', 
             linestyle='--')

    plt.title('Test set \n AUC: ' + str(round(100 * auc(1 - tprs, 1 - fprs), 1)))

    plt.show()
    
    return 