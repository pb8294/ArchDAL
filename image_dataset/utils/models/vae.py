import pickle
import sys
sys.path.append('utils/')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.models.CNN import AdaptiveBlockEncoder as AdaptiveBlock_C, AdaptiveBlockDecoder as AdaptiveBlock_CRev
from utils.layers.layers import ConvBlock, TransConvBlock

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.classifier = nn.Linear(64, 10)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def train_vae(self, x):
        # encoding
        # print(x.shape)
        x = F.relu(self.enc1(x))
        # print(x.shape)
        x = F.relu(self.enc2(x))
        # print(x.shape)
        x = F.relu(self.enc3(x))
        # print(x.shape)
        x = F.relu(self.enc4(x))
        # print(x.shape)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # print(x.shape)
        # print(hidden.shape)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print('z', z.shape)
        z = self.fc2(z)
        # print(z.shape)
        z = z.view(-1, 64, 1, 1)
        # print(z.shape)
 
        # decoding
        x = F.relu(self.dec1(z))
        # print(x.shape)
        x = F.relu(self.dec2(x))
        # print(x.shape)
        x = F.relu(self.dec3(x))
        # print(x.shape)
        reconstruction = torch.sigmoid(self.dec4(x))
        # print(reconstruction.shape)
        return reconstruction, mu, log_var
    
    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.enc1(x))
            # print(x.shape)
            x = F.relu(self.enc2(x))
            # print(x.shape)
            x = F.relu(self.enc3(x))
            # print(x.shape)
            x = F.relu(self.enc4(x))
            # print(x.shape)
            batch, _, _, _ = x.shape
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
            hidden = self.fc1(x)
            # print(x.shape)
            # print(hidden.shape)
            # get `mu` and `log_var`
            mu = self.fc_mu(hidden)
            log_var = self.fc_log_var(hidden)
            # get the latent vector through reparameterization
            z = self.reparameterize(mu, log_var)
            # print('z', z.shape)
            z = self.fc2(z)
        output = self.classifier(z)
        
class vae_conv_bottleneck(nn.Module):
    def __init__(self, input_channel, hidden_channel=64, feature_dim=64*26*26, zDim=256, truncation=3, device=torch.device("cuda:0")):
        super(vae_conv_bottleneck, self).__init__()
        self.input_dim = input_channel
        self.hidden_dim = hidden_channel
        self.feature_dim = feature_dim
        self.latent_dim = zDim
        self.device = device
        
        self.encConv1 = ConvBlock(self.input_dim, 64, kernel_size=5, pool=False, residual=False)
        self.encConv2 = AdaptiveBlock_C(64, 64, truncation=truncation, device=device)
        
        self.encFC1 = nn.Linear(self.feature_dim, self.latent_dim)
        self.encFC2 = nn.Linear(self.feature_dim, self.latent_dim)

        self.decFC1 = nn.Linear(self.latent_dim, self.feature_dim)
        
        # self.decConv1 = NonAdaptiveBlock(64, 64, truncation=truncation)
        self.decConv1 = AdaptiveBlock_CRev(64, 64, truncation=truncation, device=device)
        self.decConv2 = TransConvBlock(64, self.input_dim, stride=1, kernel_size=5, padding=1, pool=False, residual=False)
        self.classifier = nn.Linear(self.latent_dim, 10)
    
    def encode(self, x, num_beta_samples=5):
        # print('input', x.shape)
        x = self.encConv1(x)
        # print('encConv1', x.shape)
        x = self.encConv2(x)
        # print('encConv2', x.shape)
        x = x.view(-1, self.feature_dim)
        # print('reshape', x.shape)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        # print('mu', mu.shape, 'logVar', logVar.shape)
        return mu, logVar
    
    def decode(self, z, num_beta_samples=5):
        # mask, layers = self.encConv2.get_network_mask(n_samples=num_beta_samples)
        # print(mask.shape, layers)
        # print(mask[:, 0, :])
        # mask = torch.flip(mask, dims=(2,))
        # print(mask[:, 0, :])
        # print("dec input", z.shape)
        x = self.decFC1(z)
        # print("decFC1", x.shape)
        x = x.view(-1, 64, 26, 26)
        # print("reshape", x.shape)
        # x = self.decConv1(x)#, threshold=layers, mask=mask)
        Z, pi, n_layers, _ = self.encConv2.architecture_sampler(num_samples=num_beta_samples)
        x = self.decConv1(x, num_samples=5, reverse=True, Z=Z, n_layers=n_layers)
        # x = self.decConv1(x, num_samples=5, reverse=True, sampler=self.encConv2.architecture_sampler)
        # print("decConv1", x.shape)
        x = torch.sigmoid(self.decConv2(x))
        # print("recon", x.shape)
        return x
    
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar * 0.5) 
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, mu, logvar, z
        
    def classify(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.classifier(z)
        
    def get_embedding_dim(self):
        return self.hidden_dim
    
    def estimate_ELBO(self, logvar, mu, act_vec, y, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        E_loglike = F.binary_cross_entropy(act_vec, y, size_average=False)
        kl_beta_bernoulli = self.get_kl_architecture()
        kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
        ELBO = E_loglike + kl + kl_beta_bernoulli
        return ELBO, kl, kl_beta_bernoulli, E_loglike

    def get_kl_architecture(self):
        return self.encConv2.architecture_sampler.get_kl()
   