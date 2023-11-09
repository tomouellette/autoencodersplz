import torch
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


class VisionTransformerPredictor(nn.Module):
    """
    Lightweight ViT Predictor Module that predicts target blocks from context patches.
    """
    def __init__(self, 
                 num_patches,
                 embed_dim: int = 768, 
                 embed_dim_predictor: int = 384,
                 depth: int = 6,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer = nn.LayerNorm,
                 init_std = 0.02,
                 **kwargs
                 ):
        super().__init__()
        
        self.predictor_embed = nn.Linear(embed_dim, embed_dim_predictor, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim_predictor))
        depth_decay_rule = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.pos_embed_predictor = nn.Parameter(torch.randn(1, num_patches, embed_dim_predictor) * .02)
        self.predictor_blocks = nn.ModuleList([
            Block(dim = embed_dim_predictor,
                  num_heads = num_heads,
                  mlp_ratio = mlp_ratio,
                  qkv_bias = qkv_bias,
                  proj_drop = drop_rate,
                  attn_drop = attn_drop_rate,
                  drop_path = depth_decay_rule[i],
                  norm_layer = norm_layer
                  )
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(embed_dim_predictor)
        self.predictor_proj = nn.Linear(embed_dim_predictor, embed_dim, bias=True)
        
        nn.init.trunc_normal_(self.mask_token, std=init_std)
        
    def _apply_mask(self, x, mask):
        """
        Selects patches from x according to mask.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, block_size, embed_dim)
            masks (torch.Tensor): Tensors containing indices of patches in block_size to keep, (sqrt(block_size), sqrt(block_size))

        Returns:
            torch.Tensor: Input tensor with mask tokens added, of shape (batch_size, len(index), embed_dim)
        """
        # Flatten mask to 1D
        flat_mask = mask.view(-1)
        # Get indices of mask tokens
        index = flat_mask.nonzero().squeeze()
        # Gather the selected patches using the index tensor
        selected_x = torch.index_select(x, dim=1, index=index)
        
        return selected_x
    
        
    def forward(self, context_encoding, mask, target_idx):
        """
        Forward function of the ViT Predictor Module.

        Args:
            context_encoding (torch.Tensor): Tensor of shape (batch_size, context_block_size, embed_dim)
            mask (torch.Tensor): Tensor of shape (num_targets + 1, img_size//patch_size, img_size//patch_size);
                mask[0] is the mask for the context block, mask[1:] are the masks for the target blocks
                0 indicates background, 1 indicates content
            target_idx (int): Index of the target block to predict

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 1, embed_dim) containing the predicted target block
        """
        
        # Get number of batches
        num_batches = context_encoding.shape[0]
        context_block_size = context_encoding.shape[1]
        
        # Map from encoder-dim to pedictor-dim
        context_encoding = self.predictor_embed(context_encoding)
        
        # Add positional embedding to context tokens
        context_pos_embed = self.pos_embed_predictor.repeat(num_batches, 1, 1)
        context_encoding += self._apply_mask(context_pos_embed, mask[0])
        
        # Concatenate mask tokens to context tokens
        target_pos_embed = self.pos_embed_predictor.repeat(num_batches, 1, 1)
        target_pos_embed = self._apply_mask(target_pos_embed, mask[target_idx+1])
        
        pred_tokens = self.mask_token.repeat(target_pos_embed.size(0), target_pos_embed.size(1), 1)
        pred_tokens += target_pos_embed
        
        prediction_encoding = context_encoding
        prediction_encoding = torch.cat([prediction_encoding, pred_tokens], dim = 1)
        
        # Forward propagation
        for blk in self.predictor_blocks:
            prediction_encoding = blk(prediction_encoding)
        prediction_encoding = self.predictor_norm(prediction_encoding)
        
        # Return predictions for mask tokens (last len(target_masks) tokens)
        predictions = prediction_encoding[:, context_block_size:, :]
        
        # Map from predictor-dim to encoder-dim
        predictions = self.predictor_proj(predictions)
        
        return predictions