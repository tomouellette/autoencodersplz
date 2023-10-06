import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from typing import Union

class MLPMixer(nn.Module):
    """An all MLP architecture for patch-based image modeling

    Parameters
    ----------
    img_size : Union[tuple, int]
        Size of the image. If tuple, it must be (height, width), defaults to 224
    in_chans : int
        Number of channels of the input image, default to 3
    patch_size : int
        Size of the patch, defaults to 16
    dim : int
        Embed patches to a vector of size dim, defaults to 512
    depth : int
        Number of mixer layers, defaults to 12
    num_classes : int
        Number of classes, defaults to 100
    expansion_factor : int
        Expansion factor for the channels mixer, defaults to 4
    expansion_factor_token : float
        Expansion factor token that scales down dimension of patches mixer, defaults to 0.5
    dropout : float
        Dropout rate, defaults to 0.

    References
    ----------
    1. I. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, 
       A. Steiner, D. Keysers, J. Uszkoreit, M. Lucic, A. Dosovitskiy. "MLP-Mixer: An all-MLP 
       Architecture for Vision". https://arxiv.org/abs/2105.01601 2021.
    2. https://github.com/lucidrains/mlp-mixer-pytorch: Implementation adapted from lucidrains
       mlp-mixer implementation.
    """

    def __init__(
        self, 
        img_size: Union[tuple, int] = 224, 
        in_chans: int = 3, 
        patch_size: int = 16, 
        dim: int = 512, 
        depth: int = 12, 
        num_classes: int = 100, 
        expansion_factor: int = 4, 
        expansion_factor_token: float = 0.5, 
        dropout: float = 0.
    ):
        super().__init__()
        self.arguments = locals()
        image_h, image_w = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        self.num_patches = (image_h // patch_size) * (image_w // patch_size)
        
        channels_layer, tokens_layer = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patch_embed = nn.Linear((patch_size ** 2) * in_chans, dim)
        
        self.mixer_layers = nn.ModuleList([])
        for _ in range(depth):
            channels_mixer = [
                nn.LayerNorm(dim),
                channels_layer(self.num_patches, int(self.num_patches * expansion_factor)),
                nn.GELU(),
                nn.Dropout(dropout),
                channels_layer(int(self.num_patches * expansion_factor), self.num_patches),
                nn.Dropout(dropout)
            ]
            
            tokens_mixer = [
                nn.LayerNorm(dim),
                tokens_layer(dim, int(dim * expansion_factor_token)),
                nn.GELU(),
                nn.Dropout(dropout),
                tokens_layer(int(dim * expansion_factor_token), dim),
                nn.Dropout(dropout)
            ]

            self.mixer_layers.append(nn.Sequential(*channels_mixer, *tokens_mixer))
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches"""
        x = self.rearrange(x)
        x = self.patch_embed(x)
        return x

    def forward_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Mix channels and patches"""
        for mixer in self.mixer_layers:
            x = mixer(x) + x
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_embed(x)
        x = self.forward_mix(x)
        x = self.head(x)
        return x