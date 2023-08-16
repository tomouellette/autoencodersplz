import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from ..backbones.resnet import ResNet, InvertedResNet
from ..layers.dimensions import to_tuple
from ..layers.vector_quantize import VectorQuantizer

class VQVAE(nn.Module):
    """
    A vector-quantized variational autoencoder with a resnet backbone/encoder
    
    Args:
        img_size (tuple): size of input matrix
        in_chans (int): number of channels in input image
        channels (list): number of channels in each layer of the resnet encoder
        blocks (list): number of residual blocks in each layer of the resnet encoder
        codebook_dim (int): number of vectors in the codebook
        code_dim (int): dimensionality of the vectors in the codebook
        codebook_init (str): initialization strategy for setting code vectors in codebook
        metric (str): the distance metric to use for finding nearest neighbours in codebook
        beta (float): specifies the commitment loss for the embedding loss for vector quantization
        upsample_mode (str): image upsampling mode for decoder
        device (str): device to run model on (e.g. 'cpu', 'cuda:0')
    
    """
    def __init__(
        self, 
        img_size: Union[Tuple[int, int], int] = 224,
        in_chans: int = 3,
        channels: list = [64,128,256,512], 
        blocks: list = [2, 2, 2, 2], 
        codebook_dim: int = 32,
        code_dim: int = 64,
        codebook_init: str = 'uniform',
        metric: str = 'euclidean',
        beta: float = 0.25,
        upsample_mode: str = 'nearest',
        device: Optional[str] = None
    ):
        super(VQVAE, self).__init__()
        self.arguments = locals()
        img_size = to_tuple(img_size)

        if isinstance(device, type(None)):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.in_chans = in_chans
        
        # representations z|x
        self.encoder = ResNet((in_chans, *img_size), channels=channels, blocks=blocks)
        
        # quantization z_q|z
        latent_dim, _, _ = self.encoder.output_dim
        self.vector_quantize = VectorQuantizer(
            latent_dim = latent_dim,
            codebook_dim = codebook_dim,
            code_dim = code_dim,
            codebook_init = codebook_init,
            metric = metric,
            beta = beta
        )

        # decoding x|z
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

        I/O: (N, C, H, W) -> (2, z_e, h_z, w_z)
        """        
        z = self.encoder(x)
        return z
    
    def forward_quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the latent representation (z -> z_q)

        I/O: (N, latent_dim) -> (N, latent_dim), (1,)
        """        
        loss_vq, z_q = self.vector_quantize(z)
        return loss_vq, z_q
    
    def forward_decoder(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation into the original space (z -> x)

        I/O: (N, latent_dim) -> (N, C, H, W)
        """        
        xhat = self.decoder(z_q)
        return xhat
    
    def forward_loss(self, x: torch.Tensor, xhat: torch.Tensor, loss_vq: torch.Tensor) -> torch.Tensor:
        """
        Compute the ELBO = E[log(p(x|z))] - KLD(q(z|x) || p(z)) and reconstruction p(x'|z) loss
        """
        # reconstruction loss L(x, x_reconstruct)
        loss_r = F.mse_loss(xhat, x)

        # total loss
        loss = loss_r + loss_vq

        return loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, C, H, W) or ((N, C, H, W), (N, latent_dim))
        """
        z = self.forward_encoder(x)

        loss_vq, z_q = self.forward_quantize(z)

        xhat = self.forward_decoder(z_q)

        loss = self.forward_loss(x, xhat, loss_vq)
        
        return loss, xhat