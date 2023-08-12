import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from ..backbones.resnet import ResNet, InvertedResNet
from ..layers.dimensions import to_tuple

class ConvResidualAE(nn.Module):
    """
    A determinstic or variational autoencoder with a resnet backbone/encoder
    
    Args:
        img_size (tuple): size of input matrix
        in_chans (int): number of channels in input image
        channels (list): number of channels in each layer of the resnet encoder
        blocks (list): number of residual blocks in each layer of the resnet encoder
        latent_dim (int): size of latent space
        beta (float): if beta is a positive non-zero value, then model converts from deterministic to variational autoencoder
        kld_weight (float): optional value specifying additional weight on beta term
        max_temperature (int): # of loss updates until beta term reaches max value (i.e. temperature annealing of KLD loss with iter/max_temperature)
        upsample_mode (str): image upsampling mode for decoder
        device (str): device to run model on (e.g. 'cpu', 'cuda:0')
    
    """
    def __init__(
        self, 
        img_size: Union[Tuple[int, int], int] = 224,
        in_chans: int = 3,
        channels: list = [64,128,256,512], 
        blocks: list = [2, 2, 2, 2], 
        latent_dim: int = 16,
        beta: float = 0.1,
        kld_weight: Optional[float] = None,
        max_temperature: int = 1000,
        upsample_mode: str = 'nearest',
        device: Optional[str] = None
    ):
        super(ConvResidualAE, self).__init__()        
        self.arguments = locals()
        img_size = to_tuple(img_size)

        if isinstance(device, type(None)):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.iter = 0
        self.in_chans = in_chans        
        self.max_temperature = max_temperature
        
        if not isinstance(kld_weight, float):
            self.kld_weight = beta * latent_dim / math.prod(img_size)
        else:
            self.kld_weight = beta * kld_weight
        
        # representations z|x
        self.encoder = ResNet((in_chans, *img_size), channels=channels, blocks=blocks)
        
        # latent space z
        self.latent_mu = nn.Linear(math.prod(self.encoder.output_dim), latent_dim)
        self.latent_var = nn.Linear(math.prod(self.encoder.output_dim), latent_dim)

        # decoding x|z
        self.decoder_input = nn.Linear(latent_dim, math.prod(self.encoder.output_dim))
        self.decoder = InvertedResNet(
            img_size = self.encoder.output_dim,
            output_chans = in_chans,
            channels = channels[::-1],
            blocks = blocks[::-1],
            upsample_mode = upsample_mode
        )        
    
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data into a latent space (x -> z)

        I/O: (N, C, H, W) -> (2, N, latent_dim)
        """        
        z = self.encoder(x).flatten(1)        
        mu = self.latent_mu(z)
        var = self.latent_var(z)        
        return mu, var

    def forward_decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation into the original space (z -> x)

        I/O: (N, latent_dim) -> (N, C, H, W)
        """
        z = self.decoder_input(z).view(-1, *self.encoder.output_dim)
        xhat = self.decoder(z)
        return xhat
    
    def _reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to enable backpropagation through random/stochastic variable
        """
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward_loss(self, x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Compute the ELBO = E[log(p(x|z))] - KLD(q(z|x) || p(z)) and reconstruction p(x'|z) loss
        """
        self.iter += 1
        
        # reconstruction loss L(x, x_reconstruct)
        loss_r = F.mse_loss(
            xhat.flatten(1),
            x.flatten(1),
            reduction='none'
        ).sum(dim=-1)
        
        # KLD loss E[log(p(x|z))] - KLD(q(z|x) || p(z))
        if self.arguments['beta'] > 0:
            loss_kld = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=-1)
            temperature = torch.clamp(torch.Tensor([self.iter/self.max_temperature]), 0, 1)
        else:
            loss_kld = 0
            temperature = 0
        
        # beta-VAE loss (beta -> 0 is deterministic autoencoder)        
        loss = (loss_r + temperature * self.kld_weight * loss_kld).mean(dim=0)

        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, C, H, W) or ((N, C, H, W), (N, latent_dim))
        """
        mu, var = self.forward_encoder(x)
        
        if self.arguments['beta'] > 0:
            z = self._reparameterize(mu, var)
        else:
            z = mu
        
        xhat = self.forward_decoder(z)
        
        loss = self.forward_loss(x, xhat, mu, var)
        
        return loss, xhat
