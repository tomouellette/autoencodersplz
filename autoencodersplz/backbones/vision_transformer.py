import torch.nn as nn
from functools import partial
from typing import Optional, Callable, Union, Tuple
from autoencodersplz.layers.patch_embed import PatchEmbed

from timm.models import VisionTransformer
from timm.models.vision_transformer import Block, Mlp

def vision_transformer(
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1,
        global_pool: str = '',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: str = '',
        norm_layer: Optional[Callable] = None,
        patch_norm_layer: Optional[Callable] = None,
        post_norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        *args,
        **kwargs
    ):
    """
    Modification of the timm VisionTransformer class for custom patch embedding function

    Args:
        img_size: input image size
        patch_size: patch size
        in_chans: number of image input channels
        num_classes: number of classes for classification head
        global_pool: type of global pooling for final sequence (default: 'token')
        embed_dim: transformer embedding dimension
        depth: depth of transformer
        num_heads: number of attention heads
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: enable bias for qkv projections if True
        init_values: layer-scale init values (layer-scale enabled if not None)
        class_token: use class token
        fc_norm: pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'
        drop_rate: head dropout rate
        pos_drop_rate: position embedding dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: stochastic depth rate
        weight_init: weight initialization scheme
        embed_layer: patch embedding layer
        norm_layer: normalization layer
        patch_norm_layer: patch normalization layer
        post_norm_layer: post normalization layer
        act_layer: mlp activation layer
        block_fn: transformer block layer
    
    """
    arguments = locals()

    vit = VisionTransformer(
        img_size = img_size,
        patch_size = patch_size,
        in_chans = in_chans,
        embed_dim = embed_dim,
        depth = depth,
        num_heads = num_heads,
        mlp_ratio = mlp_ratio,
        pre_norm = pre_norm,
        norm_layer = norm_layer,
        qkv_bias = qkv_bias,
        qk_norm = qk_norm,
        init_values = init_values,
        class_token = class_token,
        no_embed_class = no_embed_class,
        num_classes = num_classes,
        global_pool = global_pool,
        fc_norm = fc_norm,
        drop_rate = drop_rate,
        pos_drop_rate = pos_drop_rate,
        patch_drop_rate = patch_drop_rate,
        proj_drop_rate = proj_drop_rate,
        attn_drop_rate = attn_drop_rate,
        drop_path_rate = drop_path_rate,
        weight_init = weight_init,
        act_layer = act_layer,
        block_fn = block_fn,
        mlp_layer = mlp_layer,
    )

    vit.patch_embed  = PatchEmbed(
        img_size = img_size, 
        patch_size = patch_size, 
        in_chans = in_chans, 
        embed_dim = embed_dim,
        patch_norm_layer = patch_norm_layer or partial(nn.LayerNorm, eps=1e-6),
        post_norm_layer = post_norm_layer or partial(nn.LayerNorm, eps=1e-6),
        bias = not pre_norm
    )

    vit.arguments = arguments

    return vit