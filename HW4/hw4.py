import pickle
import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from scipy.special import expit as sigmoid
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Problem 5: Neural Networks on XOR

class XORNet(nn.Module):
    def __init__(self):
        """
        Initialize the layers of your neural network

        You should use nn.Linear
        """
        super(XORNet, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)
    
    def set_l1(self, w, b):
        """
        Set the weights and bias of your first layer
        @param w: (2,2) torch tensor
        @param b: (2,) torch tensor
        """
        self.l1.weight.data = w
        self.l1.bias.data = b
    
    def set_l2(self, w, b):
        """
        Set the weights and bias of your second layer
        @param w: (1,2) torch tensor
        @param b: (1,) torch tensor
        """
        self.l2.weight.data = w
        self.l2.bias.data = b

    def forward(self, xb):
        """
        Compute a forward pass in your network.  Note that the nonlinearity should be F.relu.
        @param xb: The (n, 2) torch tensor input to your model
        @return: an (n, 1) torch tensor
        """
        hidden = F.relu(self.l1(xb))
        out = self.l2(hidden)
        return out


def fit(net, optimizer,  X, Y, n_epochs):
    """ Fit a net with BCEWithLogitsLoss.  Use the full batch size.
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param X: an (N, D) torch tensor
    @param Y: an (N, 1) torch tensor
    @param n_epochs: int, the number of epochs of training
    @return epoch_loss: Array of losses at the beginning and after each epoch. Ensure len(epoch_loss) == n_epochs+1
    """
    loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss expects raw logits
    epoch_loss = []

    # Ensure the network is in training mode.
    net.train()

    # Compute initial loss before training starts.
    with torch.no_grad():
        initial_loss = loss_fn(net(X), Y).item()
    epoch_loss.append(initial_loss)

    for epoch in range(n_epochs):
        optimizer.zero_grad()        # Clear gradients
        outputs = net(X)             # Forward pass
        loss = loss_fn(outputs, Y)   # Compute loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update parameters

        # Record loss after this epoch.
        epoch_loss.append(loss.item())

    return epoch_loss

# Problem 4: Gradient Boosting

# Reference for RegressionTree implementation
class RegressionTree:
    """
    A wrapper for the DecisionTreeRegressor from scikit-learn.
    """
    def __init__(self):
        self.tree = DecisionTreeRegressor(max_depth=3)
    
    def fit(self, X, y):
        """
        Fit the regression tree on the given data.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target values.
        """
        self.tree.fit(X, y)
    
    def predict(self, X):
        """
        Predict using the fitted regression tree.
        
        Parameters:
        X (array-like): Feature matrix.
        
        Returns:
        array-like: Predicted values.
        """
        return self.tree.predict(X)

# TODO: complete compute_functional_gradient and fit_next_tree functions in this class
class gradient_boosted_trees:
    """
    Gradient Boosted Trees for regression or binary classification using a RegressionTree learner.
    """
    def __init__(self, step_size, number_of_trees, loss_function):
        """
        Initialize the gradient boosted trees model.
        
        Parameters:
        step_size (float): The constant step size (learning rate) for updating predictions.
        number_of_trees (int): The total number of trees to use in boosting.
        loss_function (str): The loss function to use; either "least_squares" or "logistic_regression".
        """
        self.step_size = step_size
        self.number_of_trees = number_of_trees
        self.loss_function = loss_function  
        self.trees = []

    def compute_functional_gradient(self, y, f_old):
        """
        Compute the negative functional gradient of the loss function.
        
        Parameters:
        y (array-like): True target values.
        f_old (array-like): Current predictions  (using the tree ensemble already constructed).
        
        Returns:
        array-like: The computed negative functional gradient.
        """

        if self.loss_function == "least_squares":
            # For least squares: negative gradient is y - f_old.
            return y - f_old
        elif self.loss_function == "logistic_regression":
            # For logistic regression: negative gradient is y - sigmoid(f_old)
            return y - sigmoid(f_old)
        else:
            raise ValueError("Invalid loss function specified.")

    def fit_next_tree(self, X, y, f_old):
        """
        Fit the next tree on the negative functional gradient and update predictions.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): True target values.
        f_old (array-like): Current predictions (using the tree ensemble already constructed).
        
        Returns:
        array-like: Updated predictions after adding the contribution of the fitted tree.
        """
        
        # Compute the negative functional gradient (residuals)
        neg_grad = self.compute_functional_gradient(y, f_old)
        # Fit a new RegressionTree on these residuals
        tree = RegressionTree()
        tree.fit(X, neg_grad)
        # Update predictions: new prediction f_new = f_old + step_size * tree_prediction
        f_new = f_old + self.step_size * tree.predict(X)
        # Save the fitted tree into the ensemble
        self.trees.append(tree)
        return f_new


    def predict(self, X, num_trees=None):
        """
        Make predictions using the gradient boosted trees model.
        
        Parameters:
        X (array-like): Feature matrix.
        num_trees (int, optional): Number of trees to use for prediction. If None, use all trees.
        
        Returns:
        array-like: The tree ensemble prediction.
        """
        if num_trees is None:
            num_trees = len(self.trees)
        f = np.full((X.shape[0],), 0.0)
        for i in range(num_trees):
            f += self.step_size * self.trees[i].predict(X)
        return f

    def fit(self, X, y):
        """
        Fit the gradient boosted trees model on the given data.
        
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target values.
        """
        f = np.full(y.shape, 0.0)
        for i in range(self.number_of_trees):
            f = self.fit_next_tree(X, y, f)

def cross_validation(X, y, loss_function, step_sizes, number_of_trees, kf):
    """
    Perform cross-validation to compute the average accuracy for each candidate step size.
    
    This function accepts a pre-generated cross-validation splitter (such as a KFold instance)
    to avoid additional randomness in the splitting process.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target values.
    loss_function (str): Loss function to use ("least_squares" or "logistic_regression").
    step_sizes (list): List of candidate step sizes.
    number_of_trees (int): Number of trees to use in boosting.
    kf: A cross-validation splitter instance (e.g., KFold) created outside the function.
    
    Returns:
    cv_accuracies (list): A list of average cross-validation accuracies corresponding to each step size.
    """

    cv_accuracies = []
    
    for step_size in step_sizes:
        fold_accuracies = []
        for train_index, val_index in kf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Initialize and train the model on the training fold
            model = gradient_boosted_trees(step_size, number_of_trees, loss_function)
            model.fit(X_train, y_train)
            
            # Predict on the validation fold
            f_val = model.predict(X_val)
            
            # Convert continuous predictions to binary class labels:
            # For logistic regression, threshold at 0 (since sigmoid(0)=0.5)
            # For least squares, use a threshold of 0.5.
            if loss_function == "logistic_regression":
                preds = (f_val >= 0).astype(int)
            else:  # least_squares
                preds = (f_val >= 0.5).astype(int)
            
            # Compute and store the accuracy for this fold
            acc = accuracy_score(y_val, preds)
            fold_accuracies.append(acc)
        
        # Compute the average accuracy over all folds for this step size
        avg_acc = np.mean(fold_accuracies)
        cv_accuracies.append(avg_acc)
    
    return cv_accuracies 
    

def main():
    """
    Main procedure that loads training data, selects the best step_size via cross-validation,
    trains the final model, and evaluates it using the ModelTester.
    """
    # Load training data from CSV.
    # Assumes that 'train.csv' has a header row and the last column is the target.
    train_data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    
    # Candidate step sizes.
    step_sizes = [0.01, 0.1, 1, 10]

    for loss_choice in ["logistic_regression", "least_squares"]:
        print("Loss function:", loss_choice)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_accuracies = cross_validation(X_train, y_train, loss_function=loss_choice,
                                             step_sizes=step_sizes, number_of_trees=10,
                                             kf=kf)
        latex_rows = []
        for step, acc in zip(step_sizes, cv_accuracies):
            print(f"  Step size: {step:5.2f} CV Accuracy: {acc:.2f}")
            latex_rows.append(f"{step:5.2f} & {acc:.2f} \\\\")
            best_index = np.argmax(cv_accuracies)
            best_step_size = step_sizes[best_index]
            print("  Best step size: {}".format(best_step_size))
    
        # Train the final model on the full training data.
        final_model = gradient_boosted_trees(step_size=best_step_size,
                                                     number_of_trees=10,
                                                     loss_function=loss_choice)
        final_model.fit(X_train, y_train)
    
        # Save the final model to a file named after the loss function.
        model_filename = f"final_model_{loss_choice}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(final_model, f)
    
        # Print LaTeX table with step sizes, CV accuracies, and the best step size.
        print("\n\\begin{table}[h]")
        print("\\centering")
        escaped_loss_choice = loss_choice.replace('_', '\\_')
        print(f"\\caption{{Results for loss function: {escaped_loss_choice}}}")        
        print("\\begin{tabular}{cc}")
        print("\\hline")
        print("Step Size & CV Accuracy \\\\")
        print("\\hline")
        for row in latex_rows:
            print(row)
            print("\\hline")
        print(f"\\multicolumn{{2}}{{c}}{{Best step size: {best_step_size:5.2f}}} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")    
        print("\n")


def main_xor():
    """
    Train an XORNet on the XOR dataset for 5000 epochs and then use contour_torch to
    plot the resulting network's decision boundary.
    """
    # (a) Load the XOR dataset using the helper function from hw4_utils.
    X, Y = hw4_utils.XOR_data()  # X is (n, 2) and Y is (n, 1)
    
    # (b) Instantiate the XORNet and the optimizer.
    xor_net = XORNet()
    optimizer = optim.SGD(xor_net.parameters(), lr=0.1)
    
    # (c) Train for 5000 epochs using the provided fit function.
    losses = fit(xor_net, optimizer, X, Y, n_epochs=5000)
    
    # (d) Use contour_torch to visualize the decision boundary.
    # contour_torch expects a prediction function which takes a (H, W, 2) tensor as input.
    # Here, we pass our trained XORNet.
    hw4_utils.contour_torch(-1.5, 1.5, -1.5, 1.5, xor_net, ngrid=33)
    
    # Optionally, plot the loss curve.
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XORNet Training Loss (5000 epochs)")
    plt.show()


# Run this locally to generate pickle files to upload to autograder, comment out when submitting to autograder
if __name__ == "__main__":
  main()

