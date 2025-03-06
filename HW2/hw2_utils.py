import numpy as np
import torch
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_classif



## Problem Naive Bayes
def bayes_dataset(split, prefix="bayes"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1
    
    '''
    return torch.load(f"{prefix}_{split}.pth")

def bayes_eval(prefix="bayes"   ):
    import hw2
    X, y = bayes_dataset("train", prefix=prefix)
    D = hw2.bayes_MAP(X, y)
    p = hw2.bayes_MLE(y)
    Xtest, ytest = bayes_dataset("test", prefix=prefix)
    ypred = hw2.bayes_classify(D, p, Xtest)
    return ypred, ytest


## Problem Logistic Regression
def load_logistic_data():
    np.random.seed(1) # reset seed
    return linear_problem(np.array([-1.,2.]),margin=1.5,size=200)


def linear_problem(w,margin,size,bounds=[-5.,5.],trans=0.0):
    in_margin = lambda x: np.abs(w.dot(x)) / np.linalg.norm(w) < margin
    X = []
    Y = []
    for i in range(size):
        x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        while in_margin(x):
            x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        if w.dot(x) + trans > 0:
            Y.append(1.)
        else:
            Y.append(-1.)
        X.append(x)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y



## Problem Linear Regression

# Load Boston Housing Data
def load_and_split_data():
    """
    Loads the Boston Housing dataset and splits it into first half for training and second half for testing.
    Extracts 'MEDV' as the target.
    """
    # Load dataset
    data = fetch_openml(name="boston", version=1, as_frame=True)
    df = data.frame

    # Convert features and target 'MEDV' to float
    X = df.drop(columns=['MEDV']).astype(float)
    y = df['MEDV'].astype(float)

    # Split into first half (train) and second half (test)
    split_idx = len(df) // 2
    X_train, X_test = X.iloc[:split_idx, :], X.iloc[split_idx:, :]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test