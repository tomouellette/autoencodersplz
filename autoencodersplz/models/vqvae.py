import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Tuple, Union, Optional
from vector_quantize_pytorch import VectorQuantize

from ..backbones.resnet import ResNet, InvertedResNet
from ..layers.dimensions import to_tuple, collect_batch

class VQVAE(LightningModule):
    """A vector-quantized variational autoencoder with a resnet backbone/encoder
    
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
    codebook_size : int, optional
        The number of vectors in the codebook, by default 32
    codebook_dim : int, optional
        The dimension of each vector in the codebook, by default 64
    use_cosine_sim : bool, optional
        Use cosine similarity instead of euclidean distance, by default False
    kmeans_init : bool, optional
        Use kmeans to initialize the codebook, by default False
    commitment_weight : float, optional
        The weight of the commitment loss term, by default 0.5
    upsample_mode : str, optional
        The mode of upsampling, by default 'nearest'
    vq_kwargs : dict, optional
        Additional keyword arguments for the vector quantization layer; see the
        vector-quantize-pytorch package for more details on available parameters
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
    1. A. Van den Oord, O. Vinyals, K. Kavukcuoglu, "Neural Discrete Representation 
       Learning", https://arxiv.org/abs/1711.00937. NeurIPS 2017.
    2. https://github.com/lucidrains/vector-quantize-pytorch: The vector quantizer
       was sourced from a feature-rich vector quantization package by lucidrains
    """

    def __init__(
        self, 
        img_size: Union[Tuple[int, int], int] = 224,
        in_chans: int = 3,
        channels: list = [64,128,256,512],
        blocks: list = [2, 2, 2, 2],
        codebook_size: int = 256,
        codebook_dim: int = 8,
        use_cosine_sim: bool = True,
        kmeans_init: bool = True,
        commitment_weight: float = 0.5,
        upsample_mode: str = 'nearest',
        latent_dim: Optional[int] = None,
        vq_kwargs: dict = {},
        learning_rate: float = 1e-3,
        factor: float = 0.2,
        patience: int = 20,
        min_lr: float = 1e-6,
    ):
        super(VQVAE, self).__init__()
        self.arguments = locals()
        img_size = to_tuple(img_size)

        self.in_chans = in_chans
        
        # representations z|x
        self.encoder = ResNet((in_chans, *img_size), channels=channels, blocks=blocks)
        
        # quantization z_q|z
        self.latent_channels, _, _ = self.encoder.output_dim

        if isinstance(latent_dim, int):
            self.project_in = nn.Conv2d(self.latent_channels, latent_dim, kernel_size=1)
            self.project_out = nn.Conv2d(latent_dim, self.latent_channels, kernel_size=1)
        else:
            latent_dim = self.latent_channels
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.vector_quantize = VectorQuantize(
            dim = latent_dim,
            codebook_size = codebook_size,
            codebook_dim = codebook_dim,
            use_cosine_sim = use_cosine_sim,
            kmeans_init = kmeans_init,
            commitment_weight = commitment_weight,
            accept_image_fmap = True,
            **vq_kwargs
        )
        
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
        z_q, codebook, loss_vq = self.vector_quantize(z)
        z_q = self.project_out(z_q)
        return loss_vq, z_q, codebook

    
    def forward_decoder(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode the latent representation into the original space (z -> x)"""        
        xhat = self.decoder(z_q)
        return xhat
    
    def forward_loss(self, x: torch.Tensor, xhat: torch.Tensor, loss_vq: torch.Tensor) -> torch.Tensor:
        """Compute the ELBO = E[log(p(x|z))] - KLD(q(z|x) || p(z)) and reconstruction p(x'|z) loss"""
        # reconstruction loss L(x, x_reconstruct)
        loss_r = F.mse_loss(xhat, x)

        # total loss
        loss = loss_r + loss_vq

        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input images to loss and reconstruction"""
        z = self.forward_encoder(x)

        loss_vq, z_q, _ = self.forward_quantize(z)

        xhat = self.forward_decoder(z_q)

        loss = self.forward_loss(x, xhat, loss_vq)
        
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
        self.log("train_loss", loss, on_epoch=False, on_step=True, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for lightning"""
        batch = collect_batch(batch)
        loss, _ = self.forward(batch)
        self.log("val_loss", loss, on_epoch=False, on_step=True, batch_size=batch.size(0))