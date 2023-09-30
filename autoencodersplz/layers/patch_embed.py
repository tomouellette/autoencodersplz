import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import Optional, Callable, Union

from .dimensions import to_tuple

class PatchEmbed(nn.Module):
    """Patch embedding layer for 2-dimensional data (e.g. images)

    Parameters
    ----------
    img_size : Union[int, tuple, list]
        Size of input points or sequence
    patch_size : Union[int, tuple, list]
        Size of patch to embed
    embed_dim : int
        Dimensionality of embedding
    patch_norm_layer : nn.Module
        Use a pre-patch embedding normalization layer
    post_norm_layer : nn.Module
        Use a post-patch embedding normalization layer
    bias : bool
        Whether to use bias in linear projection layer
    
    References
    ----------
    1. Adapted from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py    
    """
    def __init__(
        self, 
        img_size: Union[int, tuple, list] = 224,
        patch_size: Union[int, tuple, list] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        patch_norm_layer: Optional[Callable] = nn.LayerNorm,
        post_norm_layer: Optional[Callable] = nn.LayerNorm,
        bias: bool = True
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = to_tuple(img_size)
        self.patch_size = to_tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        assert \
            self.img_size[0] % self.patch_size[0] == 0 and \
            self.img_size[1] % self.patch_size[1] == 0, \
            'img_size must be divisible by patch_size.'
        
        patch_dim = in_chans * self.patch_size[0] * self.patch_size[1]

        self.rearrange = Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=self.patch_size[0], pw=self.patch_size[1])
        self.patch_norm = patch_norm_layer(patch_dim) if patch_norm_layer else nn.Identity()
        self.proj = nn.Linear(patch_dim, embed_dim, bias=bias)
        self.post_norm = post_norm_layer(embed_dim) if post_norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.rearrange(x)
        x = self.patch_norm(x)
        x = self.proj(x)
        x = self.post_norm(x)
        return x