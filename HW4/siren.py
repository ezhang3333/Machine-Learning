import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from siren_utils import get_cameraman_tensor, get_coords, model_results
import math

ACTIVATIONS = {
    "relu": torch.relu,
    "sin": torch.sin,
    "tanh": torch.tanh
}

class SingleLayer(nn.Module):
    def __init__(self, in_features, out_features, activation, 
                 bias, is_first):
        super().__init__()
        self.is_first = is_first  # Store whether this layer is the first
        self.layer = nn.Linear(in_features, out_features, bias=bias)
        # Choose the activation function (or Identity if None)
        if activation is None:
            self.torch_activation = nn.Identity()  # no-op
        elif activation not in ACTIVATIONS:
            raise ValueError("Invalid activation")
        else:
            self.torch_activation = ACTIVATIONS[activation]
        # Set omega: for sin activation, omega is 30.0; otherwise, omega is 1.0
        self.omega = 30.0 if activation == "sin" else 1.0
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform(-1/in_features, 1/in_features)
                bound = 1.0 / self.layer.in_features
            else:
                # Other layers: uniform(-sqrt(6/in_features)/omega, sqrt(6/in_features)/omega)
                bound = math.sqrt(6 / self.layer.in_features) / self.omega
            self.layer.weight.uniform_(-bound, bound)
            if self.layer.bias is not None:
                # You can initialize bias to 0 or follow a similar rule (here we use 0)
                self.layer.bias.fill_(0.0)

    def forward(self, input):
        # Pass input through the linear layer,
        # then multiply by omega and apply the activation function.
        output = self.layer(input)
        output = output * self.omega
        output = self.torch_activation(output)
        return output

# We've implemented the model for you - you need to implement SingleLayer above
# We use 7 hidden_layer and 32 hidden_features in Siren 
#   - you do not need to experiment with different architectures, but you may.
class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, activation):
        super().__init__()

        self.net = []
        # first layer
        self.net.append(SingleLayer(in_features, hidden_features, activation,
                                    bias=True, is_first=True))
        # hidden layers
        for i in range(hidden_layers):
            self.net.append(SingleLayer(hidden_features, hidden_features, activation,
                                        bias=True, is_first=False))
        # output layer - NOTE: activation is None
        self.net.append(SingleLayer(hidden_features, out_features, activation=None, 
                                    bias=False, is_first=False))
        # combine as sequential
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # the input to this model is a batch of (x,y) pixel coordinates
        return self.net(coords)

class MyDataset(Dataset):
    def __init__(self, sidelength) -> None:
        super().__init__()
        self.sidelength = sidelength
        self.cameraman_img = get_cameraman_tensor(sidelength)  # shape: (sidelength^2, 1)
        self.coords = get_coords(sidelength)                   # shape: (sidelength^2, 2)
        # Optional: print shapes to check
        # print("Coords shape:", self.coords.shape)
        # print("Image shape:", self.cameraman_img.shape)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Return the coordinate and the corresponding pixel value.
        return self.coords[idx], self.cameraman_img[idx]
    
def train(total_epochs, batch_size, activation, hidden_size=32, hidden_layer=7):
    # Create the dataset and dataloader
    dataset = MyDataset(sidelength=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Create the Siren model using the provided SingleLayer modules
    siren_model = Siren(in_features=2, out_features=1, 
                        hidden_features=hidden_size, hidden_layers=hidden_layer, activation=activation)
    
    # Set the learning rate (try 1e-3 as a common setting)
    learning_rate = 1e-3
    # You might also experiment with different optimizers, e.g., SGD
    optim = torch.optim.Adam(lr=learning_rate, params=siren_model.parameters())
    
    losses = []  # Track losses to make plot at end
    for epoch in range(total_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch  # Unpack coordinate inputs and pixel values
            # a. Pass inputs through the Siren model
            model_output = siren_model(inputs)
            # b. Compute the Mean Squared Error loss between predictions and actual pixel values
            loss = F.mse_loss(model_output, targets)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()  # .item() to convert loss to a python scalar
        epoch_loss /= len(dataloader)
        print(f"Epoch: {epoch}, loss: {epoch_loss:4.5f}", end="\r")
        losses.append(epoch_loss)

    # Save the trained model
    torch.save(siren_model.state_dict(), f"siren_model.p")
    
    # Visualize the results:
    # model_results returns the image, gradient, and laplacian maps
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    model_output_img, grad, lap = model_results(siren_model)
    ax[0].imshow(model_output_img, cmap="gray")
    ax[1].imshow(grad, cmap="gray")
    ax[2].imshow(lap, cmap="gray")
    # Plot the training loss evolution
    ax[3].plot(losses)
    ax[3].set_title("Training Loss")
    plt.savefig(f"res_{total_epochs}_{batch_size}_{activation}.jpg")
    plt.close(fig)
    return learning_rate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Siren model.')
    parser.add_argument('-e', '--total_epochs', required=True, type=int)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-a', '--activation', required=True, choices=ACTIVATIONS.keys())
    args = parser.parse_args()
    
    train(args.total_epochs, args.batch_size, args.activation)
