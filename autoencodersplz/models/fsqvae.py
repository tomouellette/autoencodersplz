import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional
from vector_quantize_pytorch import FSQ
from lightning import LightningModule

from ..backbones.resnet import ResNet, InvertedResNet
from ..layers.dimensions import to_tuple, collect_batch

class FSQVAE(LightningModule):
    r"""A finite-scalar quantized variational autoencoder with a resnet backbone/encoder
    
    Parameters
    ----------
    img_size : Union[Tuple[int, int], int], optional
        The size of the input image, by default 224
    in_chans : int, optional
        The number of input channels, by default 3
    channels : list, optional
        The number of channels in each block of the encoder/decoder, by default [64,128,256,512]
    blocks : list, optional
        The number of blocks in each stage of the encoder/decoder, by default [2, 2, 2, 2]
    levels : list, optional
        The list of levels to generate the implicit codebook. The product of all levels is
        roughly equal to the a VQ-VAE codebook size. See section 3.3 in Ref 1.    
    latent_dim : Optional[int], optional
        If an integer is provided, the latent dimensionality (output channel size of encoder)
        will be bottlenecked/reduced to this dimension prior to quantization, by default None
    upsample_mode : str, optional
        The mode of upsampling, by default 'nearest'
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
    
    Raises
    ------
    ValueError
        If the height/width of the encoder output convolutions is less than the number of FSQ levels
    
    Notes
    -----
    To make comparison against a VQ-VAE, the FSQ paper (Ref 1, Section 4.4) suggests the following 
    FSQ levels to make the implicit codebook roughly equal to the VQ-VAE codebook dimensionality:
    | ------ | --------- | ------------ | --------------- | --------------- | ------------------ |
    | Target | 2^8 = 256 |  2^10 = 1024 |   2^12 = 4096   |  2^14 = 16384   |    2^16 = 65536    |
    | Levels | [8, 6, 5] | [8, 5, 5, 5] | [7, 5, 5, 5, 5] | [8, 8, 8, 6, 5] | [8, 8, 8, 5, 5, 5] |

    References
    ----------
    1. F. Mentzer, D. Minnen, E. Agustsson, M. Tschannen. "Finite Scalar Quantization: VQ-VAE Made
       Simple". https://arxiv.org/abs/2309.15505
    """

    def __init__(
        self, 
        img_size: Union[Tuple[int, int], int] = 224,
        in_chans: int = 3,
        channels: list = [64,128,256,512],
        blocks: list = [2, 2, 2, 2],
        levels: list = [8, 6, 5],
        latent_dim: Optional[int] = None,
        upsample_mode: str = 'nearest',
        learning_rate: float = 1e-3,
        factor: float = 0.2,
        patience: int = 20,
        min_lr: float = 1e-6,        
    ):
        super(FSQVAE, self).__init__()
        self.arguments = locals()
        img_size = to_tuple(img_size)

        self.in_chans = in_chans
        self.fsq_channels = len(levels)

        # representations z|x
        self.encoder = ResNet((in_chans, *img_size), channels=channels, blocks=blocks)
        
        # quantization z_q|z
        self.latent_channels, self.latent_h, self.latent_w = self.encoder.output_dim

        if latent_dim is None:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Conv2d(self.latent_channels, latent_dim, kernel_size=1)
            self.project_out = nn.Conv2d(latent_dim, self.latent_channels, kernel_size=1)
            self.latent_channels = latent_dim

        if self.latent_h > self.fsq_channels:
            self.to_levels = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=self.latent_h-self.fsq_channels+1)
            self.from_levels = nn.ConvTranspose2d(self.latent_channels, self.latent_channels, kernel_size=self.latent_h-self.fsq_channels+1)
        else:
            raise ValueError(
                "Height/width of encoder convolutions must be greater than or equal to " + \
                f"the number of levels (encoder channels: {self.latent_h}, FSQ channels: {self.fsq_channels})"
            )
        
        self.vector_quantize = FSQ(levels=levels)
        
        # decoding x|z
        self.decoder = InvertedResNet(
            img_size = self.encoder.output_dim,
            output_chans = in_chans,
            channels = channels[::-1],
            blocks = blocks[::-1],
            upsample_mode = upsample_mode
        )

        # lightning hyperparameters
        self.learning_rate = learning_rate
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
    
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input data into a latent space (x -> z)"""        
        z = self.encoder(x)
        return z
    
    def forward_quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize the latent representation (z -> z_q)"""
        z = self.project_in(z)
        z = self.to_levels(z)
        z_q, indices = self.vector_quantize(z)
        z_q = self.from_levels(z_q)
        z_q = self.project_out(z_q)
        return z_q, indices

    
    def forward_decoder(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation into the original space (z -> x)"""        
        xhat = self.decoder(z_q)
        return xhat
    
    def forward_loss(self, x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
        """Compute the ELBO = E[log(p(x|z))] - KLD(q(z|x) || p(z)) and reconstruction p(x'|z) loss"""
        loss = F.mse_loss(xhat, x)
        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input images to loss and reconstruction"""
        z = self.forward_encoder(x)

        z_q, _ = self.forward_quantize(z)

        xhat = self.forward_decoder(z_q)

        loss = self.forward_loss(x, xhat)
        
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
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for lightning"""
        batch = collect_batch(batch)
        loss, _ = self.forward(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)