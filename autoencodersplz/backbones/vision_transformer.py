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
    """Modification of the timm VisionTransformer class for custom patch embedding function

    Parameters
    ----------
    img_size : Union[int, Tuple[int, int]]
        Input image size, defaults to 224
    patch_size : Union[int, Tuple[int, int]]
        Patch size, defaults to 16
    in_chans : int
        Number of input channels, defaults to 3
    num_classes : int
        Number of output classes, defaults to 1
    global_pool : str
        Type of global pooling for final sequence, defaults to ''
    embed_dim : int
        Embedding dimension, defaults to 768
    depth : int
        Depth of transformer, defaults to 12
    num_heads : int
        Number of attention heads, defaults to 12
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim, defaults to 4.
    qkv_bias : bool
        Enable bias for qkv projection, defaults to True
    qk_norm : bool
        Enable qk norm, defaults to False
    init_values : Optional[float]
        Layer-scale init values, defaults to None
    class_token : bool
        Use class token, defaults to True
    no_embed_class : bool
        Do not embed class token, defaults to False
    pre_norm : bool
        Pre normalization, defaults to False
    fc_norm : Optional[bool]
        Apply normalization to fc output, defaults to None
    drop_rate : float
        Dropout rate, defaults to 0.
    pos_drop_rate : float
        Position embedding dropout rate, defaults to 0.
    patch_drop_rate : float
        Patch embedding dropout rate, defaults to 0.
    proj_drop_rate : float
        Projection dropout rate, defaults to 0.
    attn_drop_rate : float
        Attention dropout rate, defaults to 0.
    drop_path_rate : float
        Stochastic depth rate, defaults to 0.
    weight_init : str
        Weight initialization scheme, defaults to ''
    norm_layer : Optional[Callable]
        Normalization layer, defaults to None
    patch_norm_layer : Optional[Callable]
        Patch normalization layer, defaults to None
    post_norm_layer : Optional[Callable]
        Post normalization layer, defaults to None
    act_layer : Optional[Callable]
        MLP activation layer, defaults to None
    block_fn : Callable
        Transformer block layer, defaults to Block
    mlp_layer : Callable
        MLP layer, defaults to Mlp
    *args : list
        Not implemented
    **kwargs : dict
        Not implemented
    
    References
    ----------
    1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, 
       T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, 
       N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
       https://arxiv.org/abs/2010.11929. ICLR 2021.
    2. https://github.com/huggingface/pytorch-image-models: the vision transformer implementation
       was sourced from Ross Wightman's timm library.
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