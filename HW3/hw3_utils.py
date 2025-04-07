import numpy as np
import torch
import torch.utils.data
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def load_reg_data():
    # load the regression synthetic data
    np.random.seed(1) # force seed so same data is generated every time
    X = np.linspace(0,4,num=100)
    noise = np.random.normal(size=X.shape, scale=0.4)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise
    X = np.reshape(X,[-1,1])
    return X,Y

def load_xor_data():
    X = np.array([[-1,1],[1,-1],[-1,-1],[1,1]])
    Y = np.prod(X,axis=1)
    return X,Y

def linear_normal(X,Y):
    X = np.concatenate([np.ones([X.shape[0],1]),X], axis=1)
    return np.linalg.pinv(X) @ Y

def contour_plot(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) numpy ndarray as input and returns an (X,Y) numpy ndarray as output
    @param ngrid:
    """
    xgrid = np.linspace(xmin, xmax, ngrid)
    ygrid = np.linspace(ymin, ymax, ngrid)
    (xx, yy) = np.meshgrid(xgrid, ygrid)
    D = np.dstack((xx, yy))
    zz = M(D)
    C = plt.contour(xx, yy, zz,
                    cmap = 'rainbow')
    plt.clabel(C)
    plt.show()

def genlabel(x):
    y = ((x[:, 0] <= -0.5) | (x[:, 0]**2 + x[:, 1]**2 <= 1)) * 2 - 1
    return y

def gentestx(n):
    grid = np.linspace(-2.05, 2.05, n)
    X1, X2 = np.meshgrid(grid, grid)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    return X

def boundary_find(x, y):
    nn = len(y)
    n = int(np.floor(0.5 + np.sqrt(nn)))
    cond = ((y[:nn - n] * y[1:nn - n + 1] <= 0) | (y[:nn - n] * y[n:] <= 0))
    indices = np.where(cond)[0]
    return x[indices]

def boundary_plot(boundary_true, boundary_pred, tst_x, tst_y, filename, title_text):
    plt.figure()
    
    # Use a fixed random subsample of test data to avoid cluttering.
    subsample_size = 100
    if len(tst_x) > subsample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(tst_x), subsample_size, replace=False)
        tst_x_sub = tst_x[indices]
        tst_y_sub = tst_y[indices]
    else:
        tst_x_sub = tst_x
        tst_y_sub = tst_y
        
    # Plot test points: green underscore for negative labels and blue plus for positive labels.
    neg = tst_y_sub < 0
    pos = tst_y_sub > 0
    plt.scatter(tst_x_sub[neg, 0], tst_x_sub[neg, 1], color='green', marker='_', label='Test negative')
    plt.scatter(tst_x_sub[pos, 0], tst_x_sub[pos, 1], color='blue', marker='+', label='Test positive')
    
    # Plot boundaries: true boundary in black and predicted boundary in red.
    plt.scatter(boundary_true[:, 0], boundary_true[:, 1], s=10,
                color='black', label='True boundary')
    plt.scatter(boundary_pred[:, 0], boundary_pred[:, 1], s=10,
                color='red', label='Predicted boundary')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(title_text)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()