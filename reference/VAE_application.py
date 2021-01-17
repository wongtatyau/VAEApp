# import libraries
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from scipy.stats import norm

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns



class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128))
        
        self.mu     = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        
        self.latent_mapping = nn.Linear(latent_dim, 128)
        
        self.decoder = nn.Sequential(nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 28 * 28))
        
        
    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoder = self.encoder(x)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        return mu, logvar
        
    def sample_z(self, mu, logvar):
        eps = torch.rand_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
    
    def decode(self, z,x):
        latent_z = self.latent_mapping(z)
        out = self.decoder(latent_z)
        reshaped_out = torch.sigmoid(out).view(x.shape[0],1, 28,28)
        return reshaped_out
        
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        output = self.decode(z,x)
        
        return output


def show_summary(valid_dl, model):
    
    N_SAMPLES = 15
    
    model.eval()
    
    actuals, preds = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dl.dataset):
            
            actuals.append(x)
            recon_x = model(x.unsqueeze(0).to(device)).cpu()
            preds.append(recon_x.squeeze(0))
            
            if i + 1 == N_SAMPLES:
                break
                
    model.train()
    grid = make_grid([*actuals, *preds], pad_value=1, padding=1, nrow=N_SAMPLES)
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def elbo_loss(x_generated, x_true, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_generated, x_true, reduction='none').sum(dim=(1, 2, 3))
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    loss = torch.mean(kld_loss + recon_loss)
    
    return loss, torch.mean(recon_loss), torch.mean(kld_loss)


def viz_vae_clusters(vae_net):

    vae_net.to(device)
    z, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            mu, _ = vae_net.encode(x)
            z.append(mu.cpu())
            labels.append(y)
    z = torch.cat(z, dim=0)
    labels = torch.cat(labels, dim=0)

    df = pd.DataFrame({'x': z[:,0].numpy(), 
                  'y': z[:,1].numpy(),
                  'label': labels.numpy()})

    plt.figure(figsize=(15, 15))
    sns.scatterplot(x="x", y="y", hue="label", palette=sns.color_palette("Paired", 10), data=df, legend="full")
    plt.title('Vizualising Latent Vectors')


def draw_mnist_manifold_vae(model, size = 20):

    # generate a 2D point cloud of latent space using inverse CDF of Standard Gaussian.
    x_axes = norm.ppf(np.linspace(0.05, 0.95, size))
    y_axes = norm.ppf(np.linspace(0.05, 0.95, size))

    # preparing input to decoder.
    z = []
    for i, y in enumerate(x_axes):
        for j, x in enumerate(y_axes):
            z.append(torch.Tensor([x, y]))
    
    # decoding latent vectors
    z = torch.stack(z)
    
    # decoding latent vectors
    preds = model.decode(z,z).detach()
    
    # rendering a single image from predictions.
    grid = make_grid(preds, pad_value=1, padding=1, nrow=size)[0].numpy()
    
    # showing the image.
    plt.figure(figsize=(20, 20))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.title('2D Latent Space')
    plt.show()


if __name__ == "__main__":
	
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("GPU Enabled:",torch.cuda.is_available())

    # set hyperparameters
    BATCH_SIZE = 256

    if torch.cuda.is_available():
        NUM_WORKERS = 64
    else:
        NUM_WORKERS = 2

    # Training dataset and dataloader
    dataset = datasets.MNIST(
        root='../../../datasets',
        download=True,
        train = True,
        transform=transforms.ToTensor()
    )


    loader = DataLoader(
        dataset,
        num_workers=64,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Validation dataset and dataloader
    val_dataset = datasets.MNIST(
        root='../../../datasets',
        download=True,
        train = False,
        transform=transforms.ToTensor()
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=64,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    vae_net = VAE(latent_dim=2)
    opt = torch.optim.Adam(vae_net.parameters())


    loss_arr = []
    loss_rec = []
    loss_kdl = []

    # max_epochs = 60
    max_epochs = 1

    vae_net = vae_net.to(device)

    # start training
    for epoch in range(max_epochs):

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_kdl = 0.0

        for i, data in enumerate(loader, 0):

            inputs, labels = data

            inputs = inputs.to(device)

            # training steps for normal model
            opt.zero_grad()

            mu, logvar = vae_net.encode(inputs)
            z = vae_net.sample_z(mu, logvar)
            outputs = vae_net.decode(z, inputs)

            loss, recon_loss, kld_loss = elbo_loss(outputs, inputs, mu, logvar)
            loss.backward()
            opt.step()


            loss_arr.append(loss.item())
            loss_rec.append(recon_loss.item())
            loss_kdl.append(kld_loss.item())

            # print statistics
            running_loss += loss.item()
            running_loss_rec += recon_loss.item()
            running_loss_kdl += kld_loss.item()

        print('[epoch %d] loss: %.3f reconstruction loss: %.3f KLD loss: %.3f'\
              %(epoch + 1, np.mean(np.array(running_loss))/BATCH_SIZE,\
                np.mean(np.array(running_loss_rec))/BATCH_SIZE, np.mean(np.array(running_loss_kdl))/BATCH_SIZE))
        if epoch%10 == 0:
            show_summary(val_loader, vae_net)
        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_kdl = 0.0

        print("="*60)

    # draw 2-D latent variables digit clusters
    viz_vae_clusters(vae_net)

    # show the VAE generated output
    draw_mnist_manifold_vae(vae_net.cpu());