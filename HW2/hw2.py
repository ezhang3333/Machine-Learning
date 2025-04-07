import torch
import hw2_utils as utils   
from hw2_utils import load_logistic_data
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


## Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        D (2 x d Float Tensor): MAP estimation of P(X_j=1|Y=i)

    '''


    X = X.float()
    y = y.float()
    
    N, d = X.shape

    Numy0 = (y == 0).sum()
    Numy1 = (y == 1).sum()
    
    X0 = X[y == 0]
    X1 = X[y == 1]
    alpha = 1.0  
    X0_sum = X0.sum(dim=0)
    X1_sum = X1.sum(dim=0)
    D0 = (X0_sum + alpha) / (Numy0 + 2*alpha)
    D1 = (X1_sum + alpha) / (Numy1 + 2*alpha)

    D = torch.stack([D0, D1], dim=0)

    return D    

def bayes_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''

    N = len(y)
    probY0 = (y == 0).sum().float() / N

    return probY0

def bayes_classify(D, p, X):
    '''
    Arguments:
        D (2 x d Float Tensor): returned value of `bayes_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    X = X.float()
    score = torch.log(p) + torch.sum(X * torch.log(D[0]) + (1-X) * torch.log(1-D[0]), dim=1)
    sum = torch.log(1-p) + torch.sum(X*torch.log(D[1]) + (1-X) * torch.log(1-D[1]), dim = 1)
    stack = torch.stack((score, sum), dim = 1)
    classify = torch.argmax(stack, dim=1)
    return classify


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic(X, Y, lrate=0.01, num_iter=1000):
    """
    Implements logistic regression using gradient descent.

    Parameters:
        X (N x d float): Feature matrix
        Y (N float): Labels
        lrate (float, default=0.01): Learning rate
        num_iter (int, default=1000): Number of iterations

    Returns:
        w ((d + 1) x 1 float): Parameters after gradient descent
    """
    N, d = X.shape
    X = np.hstack((np.ones((N, 1)), X))
    
    w = np.zeros((d + 1, 1)) 
    Y = Y.reshape(-1, 1)  

    for _ in range(num_iter):
        predictions = sigmoid(np.dot(X, w)) 
        gradient = (1 / N) * np.dot(X.T, predictions - Y)  
        w -= lrate * gradient  

    return w

def logistic_vs_ols():
    """
    Compares logistic regression with least squares regression, 
    plots decision boundaries, and returns the figure.

    Returns:
        Figure: The figure plotted with matplotlib
    """
    X, Y = load_logistic_data()

    w_logistic = logistic(X, Y)

    X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))  
    w_ols = np.linalg.pinv(X_augmented) @ Y  

    plt.figure(figsize=(8, 6))
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], label="Class 1", color="blue")
    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], label="Class -1", color="red")

    x1_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

    w0, w1, w2 = w_logistic.flatten()
    x2_logistic = -(w0 + w1 * x1_vals) / w2
    plt.plot(x1_vals, x2_logistic, label="Logistic Regression", color="green")

    w0_ols, w1_ols, w2_ols = w_ols.flatten()
    x2_ols = -(w0_ols + w1_ols * x1_vals) / w2_ols
    plt.plot(x1_vals, x2_ols, label="Least Squares Regression", linestyle="dashed", color="purple")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Logistic Regression vs Least Squares")
    
    return plt.gcf()

## Problem Linear Regression
#
# This function implements gradient descent for linear regression, and is provided as reference
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    n = X.shape[0]
    X = np.concatenate([np.ones([n,1]),X], axis=1)
    w = np.zeros([X.shape[1],1])
    Y = Y.reshape([-1,1])
    for t in range(num_iter):
        w = w - (1/n)*lrate * (X.T @ (X @ w - Y))
    return w[:,0]


def add_bias(X):
    """
    Add a bias term (a column of ones) as the first column to the input data.

    Parameters:
        X : pandas DataFrame of shape (num_samples, num_features)
    
    Returns:
        Xp : data matrix (pandas DataFrame) with a bias column (all ones) added as the first column, with column name 'bias'
    """
    Xp = X.copy()
    Xp.insert(0, 'bias', 1) 
    return Xp

def least_squares_regression(X_train, y_train):
    """
    Compute the closed-form least squares regression solution w
                 min_w ||X_train w - y_train ||_2^2

    Parameters:
       X_train: n x d training data (possibly containing a bias column)
       y_train: n-dimensional training target

    Returns: 
       np.array, d-dimensional weight vector w that solves the least squares problem
    """
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy().reshape(-1, 1)  
    w = np.linalg.pinv(X_train_np.T @ X_train_np) @ (X_train_np.T @ y_train_np)
    return w.flatten()

def select_top_k_weights(w, X_train, X_test, k):
    """
    Select top-k features based on the absolute value of weights, including the bias term.
    
    Parameters:
        w: d-dimensional weight vector (first element corresponds to bias)
        X_train: DataFrame (with bias column as the first column)
        X_test: DataFrame (columns matching those of X_train)
        k: number of top features to select
    Returns: (top_k_indices, top_k_features, X_train_selected, X_test_selected)
            - top_k_indices: np.array, top-k feature indices. Please arrange the indices in descending order by weight (placing the index with the highest weight first, followed by the others).
            - top_k_features: names of the selected features
            - X_train_selected, X_test_selected: DataFrames with only the selected features.
    """
    topKFeatures = list(X_train.columns[np.argsort(np.abs(w))[::-1][:k]])
    Xtraining = X_train.iloc[:, np.argsort(np.abs(w))[::-1][:k]]
    Xtesting = X_test.iloc[:, np.argsort(np.abs(w))[::-1][:k]]
    return np.argsort(np.abs(w))[::-1][:k], topKFeatures, Xtraining, Xtesting

def normalize_features(X_train, X_test):
    """
    Standardize features: fit StandardScaler on X_train and transform both X_train and X_test.
    
    Parameters:
       X_train, X_test: pandas DataFrames with the original features before scaling (without the bias term).
    Returns:
        X_train_scaled, X_test_scaled: pandas DataFrames after scaling.
    """
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_test_scaled_np = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled

def compute_condition_number(X):
    """
    Compute the condition number of matrix X, defined as the ratio of the largest 
    singular value to the smallest singular value.

    Parameters:
        X: n x d data matrix (Data Frame)

    Returns:
        float: The condition number (largest singular value / smallest singular value).
    """
    X_np = X.to_numpy() 
    singular_values = np.linalg.svd(X_np, compute_uv=False)  
    condition_number = np.max(singular_values) / np.min(singular_values)
    return condition_number
