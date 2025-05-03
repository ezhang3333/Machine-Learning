import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import skimage
from PIL import Image
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import pickle

# Problem 4: Gradient Boosting


# Problem 5: Neural Networks on XOR
def contour_torch(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without the magic

    Note- your network can be passed in as paramter M without any modification.
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) torch tensor as input and returns an (X,Y) torch tensor as output
    @param ngrid: 
    """
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        D = torch.cat((xx.reshape(ngrid, ngrid, 1), yy.reshape(ngrid, ngrid, 1)), dim = 2)
        zz = M(D)[:,:,0]
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                        cmap = 'RdYlBu')
        plt.clabel(cs)
        plt.show()

def XOR_data():
    X = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    Y = (-torch.prod(X, dim=1)+1.)/2 
    return X, Y.view(-1,1)

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img.permute(1, 2, 0).view(-1, 1)