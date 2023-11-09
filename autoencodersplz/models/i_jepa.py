import copy
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from typing import Union, Tuple, Optional, Callable, List

from ..layers.dimensions import to_tuple
from ..backbones.vision_transformer import vision_transformer, VisionTransformerPredictor

class IJEPA(nn.Module):
    """Image-based joint-embedding predictive architecture (I-JEPA)

    Parameters
    ----------
    img_size : Union[int, Tuple[int, int]], optional
        Input image size, by default 224
    patch_size : Union[int, Tuple[int, int]], optional
        Patch size, by default 16
    in_chans : int, optional
        Number of input channels, by default 3
    embed_dim : int, optional
        Transformer embedding dimension, by default 768
    depth : int, optional
        Transformer depth, by default 12
    num_heads : int, optional
        Number of attention heads, by default 12
    mlp_ratio : float, optional
        MLP ratio, by default 4.
    pre_norm : bool, optional
        Whether to apply normalization before attention, by default False
    norm_layer : Optional[Callable], optional
        Normalization layer, by default nn.LayerNorm
    patch_norm_layer : Optional[Callable], optional
        Patch normalization layer, by default nn.LayerNorm
    post_norm_layer : Optional[Callable], optional
        Post embedding normalization method, by default nn.LayerNorm
    embed_dim_predictor : int, optional
        Embedding dimension for predictor, by default 384
    predictor_depth : int, optional
        Predictor depth, by default 12
    num_targets : int, optional
        Number of targets blocks, by default 4
    target_aspect_ratio : float, optional
        Aspect ratio of target blocks, by default 0.75
    target_scale : float, optional
        Scale of target blocks, by default 0.2
    context_aspect_ratio : float, optional
        Aspect ratio of context blocks, by default 1.
    context_scale : float, optional
        Scale of context blocks, by default 0.9
    
    References
    ----------
    1. M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, 
       N. Ballas, "Self-Supervised Learning from Images with a Joint-Embedding Predictive
       Architecture". https://arxiv.org/abs/2301.08243.
    2. https://github.com/lucidrains/vector-quantize-pytorch: The finite scalar quantizer
       was sourced from a feature-rich vector quantization package by lucidrains
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        pre_norm: bool = False,
        norm_layer: Optional[Callable] = nn.LayerNorm,
        patch_norm_layer: Optional[Callable] = nn.LayerNorm,
        post_norm_layer: Optional[Callable] = nn.LayerNorm,
        embed_dim_predictor: int = 384,
        predictor_depth: int = 12,
        num_targets: int = 4,
        target_aspect_ratio: float = 0.75,
        target_scale: float = 0.2,
        context_aspect_ratio: float = 1.,
        context_scale: float = 0.9
    ):
        
        super(IJEPA, self).__init__()
        self.arguments = locals()

        self.img_size = to_tuple(img_size)
        self.patch_size = to_tuple(patch_size)
        self.in_chans = in_chans
        self.post_norm_layer = post_norm_layer
        self.embed_dim = embed_dim
        self.embed_dim_predictor = embed_dim_predictor
        
        # sampling conditions for target and context blocks
        self.num_targets = num_targets
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        
        # initialize size of target and context blocks in patches
        self.target_size_patches = (0, 0)
        self.context_size_patches = (0, 0)        
        
        # context encoder s_x|x
        self.encoder = vision_transformer(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            num_heads = num_heads,
            depth = depth,
            mlp_ratio = mlp_ratio,
            pre_norm = pre_norm,
            norm_layer = norm_layer,
            patch_norm_layer = patch_norm_layer,
            post_norm_layer = post_norm_layer,
            qkv_bias = True,
            qk_norm = False,
            class_token = True,
            act_layer = nn.GELU,
            block_fn = Block,
            mlp_layer = Mlp,
            num_classes = 10,
        )
        
        # target encoder s_y|y        
        self.target_encoder = copy.deepcopy(self.encoder)

        # predictor s'_y|s_x, m_j
        self.predictor = VisionTransformerPredictor(
            num_patches = self.encoder.patch_embed.num_patches,
            embed_dim = embed_dim,
            embed_dim_predictor = embed_dim_predictor,
            num_heads = num_heads,
            depth = predictor_depth
        )
        
        # get shape of grids and numbers of patches (i.e., number of tokens)
        self.grid_size = self.target_encoder.patch_embed.grid_size
        self.num_patches = self.target_encoder.patch_embed.num_patches
        
        # decoder to map prediction blocks back to image
        self.decoder_pred = nn.Linear(embed_dim, self.patch_size[0] * self.patch_size[1] * in_chans, bias=True)    
    
    def _get_block_size(
        self,
        grid_size: Tuple[int, int],
        aspect_ratio: float,
        scale: float
    ) -> Tuple[int, int]:
        """A helper function to get the grid size of target or context blocks.
        
        Parameters
        ----------
        grid_size : Tuple[int, int]
            Grid size of raw image in terms of patches (h, w)
        aspect_ratio : float
            Aspect ratio of target or context blocks
        scale : float
            Scale of target or context blocks
            
        Returns
        -------
        block_size : Tuple[int, int]
            Grid size of target or context blocks (h, w)
        """
        num_patches_raw = grid_size[0] * grid_size[1]
        num_patches_block = int(num_patches_raw * scale)
        
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        
        block_size = (block_h, block_w)
        
        return block_size
    
    @torch.no_grad()
    def _get_target_block(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
        """Embed and sample target blocks from image.
        
        Parameters
        ----------
        img : torch.Tensor
            Input image tensor
        
        Returns
        -------
        target_blocks : torch.Tensor
            Target blocks tensor with shape (n, num_targets, block_size, embed_dim)
        target_patches_idx : List[List[int]]
            List of indices of target patches in target blocks
        all_patches_idx : List[int]
            List of indices of all target patches
        """
        # embed patches: (n, c, h, w) -> (n, num_patches, embed_dim)
        target_tokens = self.target_encoder.patch_embed(img)
        
        # get dimensions of target blocks based on target aspect ratio and scale
        (grid_h, grid_w) = self.grid_size
        self.target_size_patches = self._get_block_size(
            grid_size = self.grid_size,
            aspect_ratio = self.target_aspect_ratio,
            scale = self.target_scale
        )

        # sample target blocks
        (block_h, block_w) = self.target_size_patches
        num_blocks_h = grid_h - block_h
        num_blocks_w = grid_w - block_w

        # initialize target blocks: (n, num_targets, block_size, embed_dim)
        target_blocks = torch.zeros((
            target_tokens.shape[0], 
            self.num_targets,
            block_h*block_w,
            target_tokens.shape[-1]
        ))

        # initialize all target patches indices: (n, num_targets, block_size)
        # return for easier removal of target patches from context
        target_patches_idx = []
        all_target_patches_idx = []
        
        for target_idx in range(self.num_targets):
            start_patch_h = torch.randint(low = 0, high = num_blocks_h + 1, size = (1,)).item()
            start_patch_w = torch.randint(low = 0, high = num_blocks_w + 1, size = (1,)).item()
            start_patch = start_patch_h * grid_w + start_patch_w
            
            # patches indices in target block
            patches = []
            for h in range(block_h):
                for w in range(block_w):
                    patch_idx = start_patch + h * grid_w + w
                    patches.append(patch_idx)
                    
                    if patch_idx not in all_target_patches_idx:
                        all_target_patches_idx.append(patch_idx)
            
            # target block
            target_patches_idx.append(patches)
            target_blocks[:, target_idx] = target_tokens[:, patches, :]
        
        return target_blocks, target_patches_idx, all_target_patches_idx
        
        
    def _get_context_block(
        self, 
        img: torch.Tensor,
        all_target_patches_idx: List[int]
    ) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
        """Sample context block from image and remove target patches from context
        
        Parameters
        ----------
        img : torch.Tensor
            Input image tensor
        all_target_patches_idx : List[int]
            List of indices of all target patches
            
        Returns
        -------
        context_blocks : torch.Tensor
            Context blocks tensor with shape (n, num_targets, block_size, embed_dim)
        context_patches_idx : List[List[int]]
            List of indices of context patches in context blocks
        all_patches_idx : List[int]
            List of indices of all context patches, essentially just a list with `context_patches_idx` 
            being its only element as all predictions are made on the same context blocks.
        """        
        # embed patches: (n, c, h, w) -> (n, num_patches, embed_dim)
        context_tokens = self.encoder.patch_embed(img)
        
        # get dimensions of context blocks based on context aspect ratio and scale
        (grid_h, grid_w) = self.grid_size
        self.context_size_patches = self._get_block_size(grid_size = self.grid_size,
                                          aspect_ratio = self.context_aspect_ratio,
                                          scale = self.context_scale)
        (block_h, block_w) = self.context_size_patches
        
        # sample context blocks
        num_blocks_h = grid_h - block_h
        num_blocks_w = grid_w - block_w
        
        # initialize context blocks: (n, 1, block_size, embed_dim)
        context_blocks = torch.zeros((context_tokens.shape[0], 
                                     1, 
                                     block_h*block_w, 
                                     context_tokens.shape[-1]))
        
        # initialize all context patches indices: (n, 1, block_size)
        # return for easier removal of target patches from context
        context_patches_idx = []
        all_context_patches_idx = []        
        
        for target_idx in range(1):
            start_patch_h = torch.randint(low = 0, high = num_blocks_h + 1, size = (1,)).item()
            start_patch_w = torch.randint(low = 0, high = num_blocks_w + 1, size = (1,)).item()
            start_patch = start_patch_h * grid_w + start_patch_w
            
            # Get patches indices in target block
            patches = []
            for h in range(block_h):
                for w in range(block_w):
                    patch_idx = start_patch + h * grid_w + w
                    patches.append(patch_idx)
                    
                    if patch_idx not in all_context_patches_idx:
                        all_context_patches_idx.append(patch_idx)
            
            # Get context block
            context_patches_idx.append(patches)
            context_blocks[:, target_idx] = context_tokens[:, patches, :]
        
        # Remove target blocks from context
        mask = [i not in all_target_patches_idx for i in all_context_patches_idx]
        context_blocks = context_blocks[:, :, mask, :]
        context_patches_idx[0] = [idx for idx in context_patches_idx[0] if idx not in all_target_patches_idx]
        all_context_patches_idx = [idx for idx in all_context_patches_idx if idx not in all_target_patches_idx]
        
        return context_blocks, context_patches_idx, all_context_patches_idx
    
    def _get_mask(
            self,
            target_patches_idx: List[List[int]],
            context_patches_idx: List[List[int]],
            pixel_level: bool = False
        ) -> torch.Tensor:
        """Generate a pixel-level mask indicating the position of the context and target blocks on the image
        
        Parameters
        ----------
        target_patches_idx : List[List[int]]
            Indices of target patches in target blocks
        context_patches_idx : List[List[int]]
            Indices of context patches in context blocks
        pixel_level : bool
            Whether to generate pixel-level mask or patch-level mask
        
        Returns
        -------
        mask : torch.Tensor
            Pixel- or patch-level mask tensor. mask[0] is the context, mask[1:] are the targets
        
        Notes
        -----
        Zero values represent background and one values represent content.
        """        
        # get image size and patch size in pixels
        img_size = self.img_size
        patch_size = self.patch_size
        num_layers = self.num_targets + 1
        
        # initialize mask tensor
        if pixel_level:
            mask = torch.zeros((num_layers, img_size[0], img_size[1]))
        else:
            mask = torch.zeros((num_layers, img_size[0] // patch_size[0], img_size[1] // patch_size[1]))
        
        # loop through context blocks
        for context_idx in range(len(context_patches_idx)):
            for context_patch_idx in context_patches_idx[context_idx]:
                if pixel_level:
                    row = (context_patch_idx // (img_size[0] // patch_size[0])) * patch_size[0]
                    col = (context_patch_idx % (img_size[1] // patch_size[1])) * patch_size[1]
                    mask[0, row:row + patch_size[0], col:col + patch_size[1]] = 1
                else:
                    mask[0, context_patch_idx // (img_size[1] // patch_size[1]), context_patch_idx % (img_size[1] // patch_size[1])] = 1
        
        # loop through target blocks
        for target_idx in range(len(target_patches_idx)):
            for target_patch_idx in target_patches_idx[target_idx]:
                if pixel_level:
                    row = (target_patch_idx // (img_size[0] // patch_size[0])) * patch_size[0]
                    col = (target_patch_idx % (img_size[1] // patch_size[1])) * patch_size[1]
                    mask[target_idx + 1, row:row + patch_size[0], col:col + patch_size[1]] = 1
                else:
                    mask[target_idx + 1, target_patch_idx // (img_size[1] // patch_size[1]), target_patch_idx % (img_size[1] // patch_size[1])] = 1
        
        return mask
    
    def _patches_to_imgs(self, decoded_tokens: torch.Tensor, size_patches: tuple) -> torch.Tensor:
        """Convert encoded patches back to image

        Parameters
        ----------
        decoded_tokens : torch.Tensor
            Encoded patches tensor with shape (n, block_size, embed_dim)
        size_patches : tuple
            Size of image in terms of patches (h, w)
        
        Returns
        -------
        pixels : torch.Tensor
            Reconstructed image tensor
        """
        pixels = rearrange(
            decoded_tokens, 
            'b (h w) (ph pw c) -> b c (h ph) (w pw)',
            h = size_patches[0],
            w = size_patches[1],
            ph = self.patch_size[0], 
            pw = self.patch_size[1],
        )
        
        return pixels
    
    def _predictions_to_img(
        self,
        prediction_blocks: torch.Tensor,
        mask: torch.Tensor,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """Decode prediction blocks to images
        
        Parameters
        ----------
        prediction_blocks : torch.Tensor
            Prediction blocks tensor with shape (n, num_targets, block_size, embed_dim)
        mask : torch.Tensor
            Pixel-level mask tensor. mask[0] is the context, mask[1:] are the targets
        img : torch.Tensor
            Input image tensor
            
        Returns
        -------
        reconstructed_img : torch.Tensor
            Reconstructed image tensor
        """        
        # predict target images from prediction blocks
        prediction_blocks_patches = self.decoder_pred(prediction_blocks)
        
        # copy raw image
        reconstructed_img = copy.deepcopy(img)
        
        # initialize an object to store each reconstructed target image -- not used for now, but could be helpful in the future
        target_imgs = torch.zeros((self.num_batch, 
                                   self.num_targets, 
                                   self.in_chans,
                                   self.target_size_patches[0] * self.patch_size[0],
                                   self.target_size_patches[1] * self.patch_size[1]))
        
        # get reconstructed target images: (n, num_targets, block_size, embed_dim) -> (n, num_targets, in_chans, height_pixels, width_pixels)
        # replace the prediction region of the raw image with the reconstructed ones
        for target_idx in range(self.num_targets):
            # convert prediction blocks to image
            target_img = self._patches_to_imgs(prediction_blocks_patches[:, target_idx, :, :], 
                                               self.target_size_patches)
            target_imgs[:, target_idx, :, :, :] = target_img
            
            # replace the prediction region of the raw image with the reconstructed ones
            reconstructed_img[:, :, mask[target_idx + 1] == 1] = rearrange(target_img, 'b c h w -> b c (h w)')
        
        return reconstructed_img        
    
    def forward_encoder(
        self, 
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[int]], List[int], torch.Tensor, List[List[int]], List[int]]:
        """Encode the target and context blocks
        
        Parameters
        ----------
        img : torch.Tensor
            Input image tensor
        
        Returns
        -------
        target_blocks : torch.Tensor
            Target blocks tensor with shape (n, num_targets, block_size, embed_dim)
        target_patches_idx : List[List[int]]
            List of indices of target patches in target blocks, with shape (n, num_targets, block_size)
        all_target_patches_idx : List[int]
            List of non-repetitive indices of all target patches
        context_blocks : torch.Tensor
            Context blocks tensor with shape (n, num_targets, block_size, embed_dim)
        context_patches_idx : List[List[int]]
            List of indices of context patches in context blocks, with shape (n, 1, block_size)
        all_context_patches_idx : List[int]
            List of non-repetitive indices of all context patches
        """
        # Get target embedding
        target_blocks, target_patches_idx, all_target_patches_idx = self._get_target_block(img = img)
        
        # Get context embedding
        context_blocks, context_patches_idx, all_context_patches_idx = self._get_context_block(img = img, all_target_patches_idx = all_target_patches_idx)
        
        return target_blocks, target_patches_idx, all_target_patches_idx, context_blocks, context_patches_idx, all_context_patches_idx
    
    def forward_decoder(
        self,
        context_blocks: torch.Tensor,
        target_blocks_shape: torch.Size,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Decode the target blocks from the context blocks
        
        Parameters
        ----------
        context_blocks : torch.Tensor
            Context blocks tensor with shape (n, 1, block_size, embed_dim)
        target_blocks_shape : torch.Size
            Shape of target blocks tensor: (n, num_targets, block_size, embed_dim)
        mask : torch.Tensor
            Patch-level mask tensor. mask[0] is the context, mask[1:] are the targets
        
        Returns
        -------
        prediction_blocks : torch.Tensor
            Prediction blocks tensor with shape (n, num_targets, block_size, embed_dim)
        """
        # Initialize prediction blocks: (n, num_targets, block_size, embed_dim)
        prediction_blocks = torch.zeros(target_blocks_shape)
        
        for target_idx in range(self.num_targets):
            # Predict target block based on context patches and target masks
            prediction_blocks[:, target_idx, :, :] = self.predictor(
                context_encoding = context_blocks[:, 0, :, :],
                mask = mask,
                target_idx = target_idx
            )
        
        return prediction_blocks    
    
    def forward_loss(
        self,
        target_blocks: torch.Tensor,
        prediction_blocks: torch.Tensor
    ) -> torch.Tensor:
        """L2 loss function
        
        Parameters
        ----------
        target_blocks : torch.Tensor
            Target blocks tensor with shape (n, num_targets, block_size, embed_dim)
        prediction_blocks : torch.Tensor
            Prediction blocks tensor with shape (n, num_targets, block_size, embed_dim)
        
        Returns
        -------
        loss : torch.Tensor
            L2 loss between target blocks and prediction blocks
        """        
        # Calculate L2 loss between target blocks and prediction blocks
        loss = F.mse_loss(prediction_blocks, target_blocks, reduction = "mean")        
        return loss    
    
    def forward(
        self,
        img: torch.Tensor
    ) -> torch.Tensor:
        """Input images to loss and reconstruction"""        
        # get info from img
        self.num_batch = img.shape[0]
        self.img_size = img.shape[2:]
        self.patch_size = torch.tensor(self.patch_size)
        
        # forward encoders (_ = all_target_patches_idx, _ = all_context_patches_idx)
        target_blocks, target_patches_idx, _, context_blocks, context_patches_idx, _ = self.forward_encoder(img)
        
        # get patch-level and pixel-level masks indicating context and targets
        patch_mask, pixel_mask = (
            self._get_mask(
                target_patches_idx = target_patches_idx,
                context_patches_idx = context_patches_idx,
                pixel_level = False
            ),
            self._get_mask(
                target_patches_idx = target_patches_idx,
                context_patches_idx = context_patches_idx,
                pixel_level = True
            ))
        
        # forward decoder
        prediction_blocks = self.forward_decoder(
            context_blocks = context_blocks,
            target_blocks_shape = target_blocks.shape,
            mask = patch_mask)        
        
        # Forward loss
        loss = self.forward_loss(target_blocks = target_blocks, prediction_blocks = prediction_blocks)
        
        return loss, self._predictions_to_img(prediction_blocks, pixel_mask, img)