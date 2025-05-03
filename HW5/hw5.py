import hw5_utils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim




## Problem Convolutional Neural Networks

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=7)
        
        # 2) Max Pool: 2×2
        self.pool  = nn.MaxPool2d(kernel_size=2)
        
        # 3) Conv2: 7→3 channels, 2×2 kernel
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=2)
        self.bn2   = nn.BatchNorm2d(num_features=3)
        
        # 4) Fully connected: 3*2*2 → 10
        self.fc    = nn.Linear(in_features=3*2*2, out_features=10)
        


    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        x = xb.unsqueeze(1)
        
        # conv1 → BN → ReLU → pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # conv2 → BN → ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # flatten and FC
        x = x.reshape(x.size(0), -1)  # (N, 12)
        x = self.fc(x)                # (N, 10)
        return x
    

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train,  batch_size=batch_size, shuffle=False)
    test_dl  = torch.utils.data.DataLoader(test,   batch_size=batch_size)

    train_losses = []
    test_losses  = []

    # -- initial losses (evaluation mode) --
    net.eval()
    with torch.no_grad():
        train_losses.append(hw5_utils.epoch_loss(net, loss_func, train_dl).item())
        test_losses.append( hw5_utils.epoch_loss(net, loss_func, test_dl).item() )

    for epoch in range(n_epochs):
        # -- training pass --
        net.train()
        for xb, yb in train_dl:
            hw5_utils.train_batch(net, loss_func, xb, yb, opt=optimizer)

        # -- evaluation pass --
        net.eval()
        with torch.no_grad():
            train_losses.append(hw5_utils.epoch_loss(net, loss_func, train_dl).item())
            test_losses.append( hw5_utils.epoch_loss(net, loss_func, test_dl).item() )

    return train_losses, test_losses




## Problem ResNet

class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        # f(x) path:
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_channels)


    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = self.conv1(x)      # Conv → same shape
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x            # skip connection
        out = F.relu(out)
        return out


## Problem Convolutional Neural Networks

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        # (i) initial conv + BN + ReLU + pool
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn   = nn.BatchNorm2d(num_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # (v) one residual block
        self.block = Block(num_channels)
        # (vi) global average pool
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # (vii) final linear
        self.fc = nn.Linear(num_channels, num_classes)


    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        #x = torch.unsqueeze(x, 1)
        x = self.conv(x)         # → (N,C,4,4)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)         # → (N,C,2,2)
        x = self.block(x)        # → (N,C,2,2)
        x = self.avgpool(x)      # → (N,C,1,1)
        x = torch.flatten(x, 1)  # → (N, C)
        x = self.fc(x)           # → (N,10)
        return x
    

