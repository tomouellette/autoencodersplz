import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Tuple, Optional, Union

from ..backbones.mlp import LinearResidualNet
from ..layers.dimensions import to_tuple, collect_batch

class LinearResidualAE(LightningModule):
    """A fully connected autoencoder with a linear residual network backbone and decoder
    
    Parameters
    ----------
    img_size : Union[Tuple[int, int], int], optional
        The size of the input image, by default 224
    in_chans : int, optional
        The number of input channels, by default 3
    hidden_dim : list, optional
        The number of hidden units in each layer of the encoder/decoder, by default [64, 64]
    blocks : list, optional
        The number of blocks in each stage of the encoder/decoder, by default [2, 2]
    dropout_rate : float, optional
        The dropout rate, by default 0.0
    with_batch_norm : bool, optional
        Whether to use batch normalization, by default False
    zero_initialization : bool, optional
        Whether to initialize weights and biases to zero, by default False
    latent_dim : int, optional
        The dimension of the latent space, by default 16
    beta : float, optional
        The weight of the KL divergence term, by default 0.1
    kld_weight : Optional[float], optional
        Additional weight on the KL divergence term, by default None
    max_temperature : int, optional
        The number of iterations/batches until the KL divergence term reaches its maximum value,
        by default 1000
    learning_rate : float, optional
        The learning rate if using pytorch lightning for training, by default 1e-3
    factor : float, optional
        The factor to reduce the learning rate by if using pytorch lightning for training,
        by default 0.2
    patience : int, optional
        The number of epochs to wait before reducing the learning rate if using pytorch lightning
        for training, by default 20
    min_lr : float, optional
        The minimum learning rate if using pytorch lightning for training, by default 1e-6
    
    References
    ----------
    1. A. Emin Orhan, X. Pitkow, "Skip Connections Eliminate Singularities"
       https://arxiv.org/abs/1701.09175. ICLR 2018.
    2. D.P. Kingma & M. Welling, "Auto-Encoding Variational Bayes". 
       https://arxiv.org/abs/1312.6114. ICLR 2014.
    """
    
    def __init__(
        self, 
        img_size: Union[Tuple[int, int], int] = 224,
        in_chans: int = 3,
        hidden_dim: list = [64, 64],
        blocks: int = [2, 2],
        dropout_rate: float = 0.0,
        with_batch_norm: bool = False,
        zero_initialization: bool = False,
        latent_dim: int = 16,
        beta: float = 0.1,
        kld_weight: Optional[float] = None,
        max_temperature: int = 1000,
        learning_rate: float = 1e-3,
        factor: float = 0.2,
        patience: int = 20,
        min_lr: float = 1e-6        
    ):
        super(LinearResidualAE, self).__init__()
        self.arguments = locals()
        self.img_size = to_tuple(img_size)
        self.iter = 0
        self.in_chans = in_chans        
        self.max_temperature = max_temperature
        
        if not isinstance(kld_weight, float):
            self.kld_weight = beta * latent_dim / math.prod(self.img_size)
        else:
            self.kld_weight = beta * kld_weight
        
        # representations z|x
        self.encoder = LinearResidualNet(
            input_dim = math.prod(self.img_size) * in_chans,
            output_dim = 2*latent_dim,
            hidden_dim = hidden_dim,
            blocks = blocks,
            dropout_rate = dropout_rate,
            with_batch_norm = with_batch_norm,
            zero_initialization = zero_initialization
        )
        
        # latent space z
        self.latent_mu = nn.Linear(2*latent_dim, latent_dim)
        self.latent_var = nn.Linear(2*latent_dim, latent_dim)

        # decoding x|z
        self.decoder_input = nn.Linear(latent_dim, 2*latent_dim)
        self.decoder = LinearResidualNet(
            input_dim = 2*latent_dim,
            output_dim = math.prod(self.img_size) * in_chans,
            hidden_dim = hidden_dim,
            blocks = blocks,
            dropout_rate = dropout_rate,
            with_batch_norm = with_batch_norm,
            zero_initialization = zero_initialization
        )

        # lightning hyperparameters
        self.learning_rate = learning_rate
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
    
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data into a latent space (x -> z)

        I/O: (N, C, H, W) -> (2, N, latent_dim)
        """        
        z = self.encoder(x.flatten(1))
        mu = self.latent_mu(z)
        var = self.latent_var(z)        
        return mu, var

    def forward_decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation into the original space (z -> x)"""
        z = self.decoder_input(z)
        xhat = self.decoder(z).view(-1, self.in_chans, *self.img_size)
        return xhat
    
    def _reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to enable backpropagation through random/stochastic variable"""
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward_loss(self, x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Compute the ELBO = E[log(p(x|z))] - KLD(q(z|x) || p(z))"""
        self.iter += 1
        
        # reconstruction loss
        loss_r = F.mse_loss(
            xhat.flatten(1),
            x.flatten(1),
            reduction='none'
        ).sum(dim=-1)
        
        # KLD loss
        if self.arguments['beta'] > 0:
            loss_kld = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=-1)
            temperature = torch.clamp(torch.Tensor([self.iter/self.max_temperature]), 0, 1).to(x.device)
        else:
            loss_kld = 0
            temperature = 0
        
        # beta-VAE loss (beta -> 0 is deterministic autoencoder)        
        loss = (loss_r + temperature * self.kld_weight * loss_kld).mean(dim=0)

        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input images to loss and reconstruction"""
        mu, var = self.forward_encoder(x)
        
        if self.arguments['beta'] > 0:
            z = self._reparameterize(mu, var)
        else:
            z = mu
        
        xhat = self.forward_decoder(z)
        
        loss = self.forward_loss(x, xhat, mu, var)
        
        return loss, xhat
    
    def configure_optimizers(self):
        """Optimization configuration for lightning"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = "min", 
            factor = self.factor, 
            patience = self.patience, 
            min_lr = self.min_lr
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        """Training step for lightning"""        
        batch = collect_batch(batch)
        loss, _ = self.forward(batch)
        self.log("train_loss", loss, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for lightning"""
        batch = collect_batch(batch)
        loss, _ = self.forward(batch)
        self.log("val_loss", loss, batch_size=batch.size(0))
