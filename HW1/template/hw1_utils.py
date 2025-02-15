#!/bin/python

import sys
import math
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import random
import torch
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights 
import faiss
from collections import Counter
import pandas as pd

def load_iris_data(ratio=1):
    """
    Loads the Iris dataset and splits it into testing and training datasets

    arguments:
    ratio -- the ratio of test set to the entire dataset

    return:
    X_train -- training dataset
    X_test -- test datasetls
    
    Y_train -- training labels
    Y_test -- test labels
    """
    X, Y = load_iris(return_X_y=True)
    return train_test_split(X, Y, test_size=1-ratio)

def line_plot(*data, min_k=2, output_file='output.pdf'):
    """
    Plots the line plot of the given data

    arguments:
    *data -- a list of data arrays to be plotted
    min_k -- the first index on the x axis
    output_file -- the location of the file to print the plot to
    """
    fig = plt.figure(figsize=(30, 10))
    for d in data:
        plt.plot(range(min_k, min_k + len(d)), d)
    plt.savefig(output_file)

def scatter_plot_2d_project(*data, output_file='output.pdf', ncol=3):
    """
    Plots the scatter plot of the given data

    arguments:
    *data -- a list of data matricies to be plotted
    output_file -- the location of the file to print the plot to

    example:
    If X is a data matrix of shape (n,d) and A is an assignment matrix of shape (n,k) and C is the matrix of centers of shape (k,d), you would write:
        scatter_plot_2d_project(X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:], C)
    """
    fig = plt.figure(figsize=(30, 10))
    d = data[0].shape[1]
    for i in range(d):
        for j in range(i):
            ax = plt.subplot(math.ceil(d*(d-1)/2/ncol), ncol, i*(i-1)//2+j+1)
            for X in data:
                plt.scatter(X[:,j], X[:,i])
            plt.xlabel('dim{}'.format(j))
            plt.ylabel('dim{}'.format(i))
    plt.savefig(output_file)

def gaussian_plot_2d_project(mu, variances, *data, output_file='output.pdf', ncol=3):
    """
    Plots the scatter plot of the given data

    arguments:
    *data -- a list of data matricies to be plotted
    output_file -- the location of the file to print the plot to

    example:
    If X is a data matrix of shape (n,d) and A is an assignment matrix of shape (n,k) and mu is the matrix of centers of shape (k,d) and variances is the variance matrix of shape(k,d), you would write:
        gaussian_plot_2d_project(mu, variances, X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:])
    """
    fig = plt.figure(figsize=(30, 10))
    d = data[0].shape[1]
    for i in range(d):
        for j in range(i):
            ax = plt.subplot(math.ceil(d*(d-1)/2/ncol), ncol, i*(i-1)//2+j+1)
            for (l, X) in enumerate(data):
                plt.scatter(X[:,j], X[:,i])
                if l < mu.shape[0] and l < variances.shape[0]:
                    (xmin, xmax, ymin, ymax) = (np.min(X[:,j]), np.max(X[:,j]), np.min(X[:,i]), np.max(X[:,i]))
                    xrange = np.arange(xmin, xmax, (xmax-xmin)/256)
                    yrange = np.arange(ymin, ymax, (ymax-ymin)/256)
                    xmesh, ymesh = np.meshgrid(xrange, yrange)
                    stack = np.dstack((xmesh, ymesh))
                    plt.contour(xmesh, ymesh, multivariate_normal.pdf(stack, mean=mu[l,[j,i]], cov=np.diag(variances[l,[j,i]])), cmap='Greys')
            plt.xlabel('dim{}'.format(j))
            plt.ylabel('dim{}'.format(i))
    plt.savefig(output_file)


# ----- Load ResNet18 Model -----
def get_resnet18():
    weights = ResNet18_Weights.DEFAULT  
    resnet18 = models.resnet18(weights=weights)  
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  
    resnet18.eval()  
    transform = weights.transforms()

    seed=42
    random.seed(seed)  
    np.random.seed(seed)
    
    return resnet18, transform

# ----- Load CIFAR-10 Subset -----
def load_cifar10_subset(transform, n_train_per_class=50, n_test=100):
    # Load CIFAR-10 training data
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Organize training data by class
    train_data, train_labels = [], []
    class_counts = {i: 0 for i in range(10)}  # Track counts per class

    for img, label in train_dataset:
        if class_counts[label] < n_train_per_class:
            train_data.append(img.numpy())  # Convert to numpy
            train_labels.append(label)
            class_counts[label] += 1
        if all(count >= n_train_per_class for count in class_counts.values()):
            break  # Stop when we have enough per class

    # Select random test subset
    test_indices = np.random.choice(len(test_dataset), n_test, replace=False)
    test_data = np.array([test_dataset[i][0].numpy() for i in test_indices])
    test_labels = np.array([test_dataset[i][1] for i in test_indices])

    return np.array(train_data), np.array(train_labels), test_data, test_labels

def identity_embedding(X):
    """Flattens CIFAR-10 images for FAISS indexing."""
    return X.reshape(X.shape[0], -1).astype(np.float32)

def resnet18_embedding(X, model=None):
    """Computes embeddings using ResNet18."""
    X_tensor = torch.tensor(X, dtype=torch.float32)

    if len(X_tensor.shape) == 3:  # Ensure batch dimension exists
        X_tensor = X_tensor.unsqueeze(0)

    if X_tensor.shape[1] != 3:
        X_tensor = X_tensor.repeat(1, 3, 1, 1)

    with torch.no_grad():
        embeddings = model(X_tensor).squeeze(-1).squeeze(-1).numpy()

    return embeddings.astype(np.float32)