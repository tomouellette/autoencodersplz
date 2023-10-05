import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from lightning import LightningModule
from typing import Union, Tuple, Optional, Callable
from timm.models.vision_transformer import Block, Mlp

from ..layers.dimensions import to_tuple, collect_batch
from ..backbones.vision_transformer import vision_transformer
from ..trainers.schedulers import CosineDecayWarmUp

class MAE(LightningModule):
    """A masked autoencoder with a vision transformer backbone/encoder and decoder

    Parameters
    ----------
    img_size : Union[int, Tuple[int, int]], optional
        The size of the input image, by default 224
    patch_size : Union[int, Tuple[int, int]], optional
        The size of the masked patches, by default 16
    in_chans : int, optional
        The number of input channels, by default 3
    mask_ratio : float, optional
        The ratio of masked patches, by default 0.5
    embed_dim : int, optional
        The dimension of the embedding, by default 768
    depth : int, optional
        The number of transformer blocks, by default 12
    num_heads : int, optional
        The number of attention heads, by default 12
    mlp_ratio : float, optional
        The ratio of the hidden dimension to the embedding dimension, by default 4
    pre_norm : bool, optional
        Whether to apply layer normalization before the attention layer, by default False
    decoder_embed_dim : int, optional
        The dimension of the decoder embedding, by default 768
    decoder_depth : int, optional
        The number of decoder transformer blocks, by default 12
    decoder_num_heads : int, optional
        The number of decoder attention heads, by default 12
    norm_layer : Optional[Callable], optional
        The normalization layer, by default nn.LayerNorm
    patch_norm_layer : Optional[Callable], optional
        The patch normalization layer, by default nn.LayerNorm
    post_norm_layer : Optional[Callable], optional
        The post-normalization layer, by default nn.LayerNorm
    learning_rate : float, optional
        The learning rate if using pytorch lightning for training, by default 1e-3
    weight_decay : float, optional
        The weight decay if using pytorch lightning for training, by default 1e-6
    betas : Tuple[float, float], optional
        The betas if using pytorch lightning for training, by default (0.9, 0.999)
    warmup_epochs : int, optional
        The number of epochs to warmup the learning rate if using pytorch lightning for training,
        by default 10
    min_lr : float, optional
        The minimum learning rate if using pytorch lightning for training, by default 1e-6
    
    References
    ----------
    1. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, "Masked Autoencoders Are
       Scalable Vision Learners". https://arxiv.org/abs/2111.06377. CVPR 2022.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        mask_ratio: float = 0.5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        pre_norm: bool = False,
        decoder_embed_dim: int = 768,
        decoder_depth: int = 12,
        decoder_num_heads: int = 12,
        norm_layer: Optional[Callable] = nn.LayerNorm,
        patch_norm_layer: Optional[Callable] = nn.LayerNorm,
        post_norm_layer: Optional[Callable] = nn.LayerNorm,
        learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        min_lr: float = 1e-6,
        warmup_epochs: int = 40,
    ):
        super(MAE, self).__init__()
        self.arguments = locals()
        
        self.mask_ratio = mask_ratio
        self.decoder_embed_dim = decoder_embed_dim

        # encoder z|x
        self.encoder = vision_transformer(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
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
            num_classes = 100,
        )

        num_patches = self.encoder.patch_embed.num_patches
        self.patch_height, self.patch_width  = to_tuple(patch_size)

        # decoder x'|z
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(decoder_embed_dim))        
        
        self.decoder_blocks = nn.Sequential(*[
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)
        ])
        
        self.decoder_pos_embed = nn.Embedding(num_patches, decoder_embed_dim)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)        
        self.decoder_pred = nn.Linear(decoder_embed_dim, math.prod([self.patch_height, self.patch_width]) * in_chans, bias=True)

        # lightning hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr                
    
    def _patches_to_img(self, decoded_tokens: torch.Tensor) -> torch.Tensor:
        pixels = rearrange(
            decoded_tokens, 
            'b (h w) (ph pw c) -> b c (h ph) (w pw)',
            h = self.img_size[0] // self.patch_height,
            w = self.img_size[1] // self.patch_width,
            ph = self.patch_height, pw = self.patch_width,
        )

        return pixels
    
    def random_masking(self, tokens: torch.Tensor, mask_ratio: float):
        device = tokens.device
        batch_size, num_patches, *_ = tokens.shape

        # collect masked and unmasked indices
        num_masked = int((1-mask_ratio) * num_patches)
        rand_ids = torch.rand(batch_size, num_patches, device=device).argsort(dim=-1)
        mask_ids, unmask_ids = rand_ids[:, :num_masked], rand_ids[:, num_masked:]

        # mask tokens
        batch_range = torch.arange(batch_size)[:, None]
        tokens = tokens[batch_range, unmask_ids]
        
        return tokens, mask_ids, unmask_ids
    
    def forward_encoder(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches: (n, c, h, w) -> (n, num_patches+1, embed_dim)
        tokens = self.encoder.patch_embed(img)

        # add pos embed w/o class token: (n, num_patches+1, embed_dim) -> (n, num_patches, embed_dim)
        tokens = tokens + self.encoder.pos_embed[:, 1:, :]

        # masking: (n, num_patches, embed_dim) -> (n, num_patches * (1-mask_ratio), embed_dim)
        tokens, mask_ids, unmask_ids = self.random_masking(tokens, self.mask_ratio)

        # transformer blocks
        tokens = self.encoder.blocks(tokens)
        tokens = self.encoder.norm(tokens)

        return tokens, mask_ids, unmask_ids
    
    def forward_decoder(
        self, 
        tokens: torch.Tensor, 
        mask_ids: torch.Tensor, 
        unmask_ids: torch.Tensor,
    ) -> torch.Tensor:
        device = tokens.device
        (batch_size, *_), num_masked, num_unmasked = tokens.shape, mask_ids.shape[1], unmask_ids.shape[1]

        #  embed tokens
        tokens = self.decoder_embed(tokens)
        
        # add pos embed w/o cls token to unmasked tokens
        unmasked_tokens = tokens + self.decoder_pos_embed(unmask_ids)

        # add pos embed w/o cls token to masked tokens
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch_size, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_embed(mask_ids)

        # construct new tensor to store masked and unmasked tokens
        decoder_tokens = torch.zeros(
            batch_size, 
            num_masked + num_unmasked, 
            self.decoder_embed_dim, 
            device = device
        )

        # combine masked tokens and unmasked decoder tokens
        batch_range = torch.arange(batch_size)[:, None]
        decoder_tokens[batch_range, unmask_ids] = unmasked_tokens
        decoder_tokens[batch_range, mask_ids] = mask_tokens

        # transformer blocks
        decoded_tokens = self.decoder_blocks(decoder_tokens)
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # predict patches
        decoded_tokens = self.decoder_pred(decoded_tokens)

        return decoded_tokens
    
    def forward_loss(
        self, 
        img: torch.Tensor, 
        decoded_tokens: torch.Tensor, 
        mask_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = img.shape[0]
        batch_range = torch.arange(batch_size)[:, None]
        
        img = self.encoder.patch_embed.rearrange(img)[batch_range, mask_ids]
        pred = decoded_tokens[batch_range, mask_ids]
        loss = F.mse_loss(pred, img, reduction='mean')
        
        return loss
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device, self.img_size = img.device, img.shape[2:]
        
        batch_size = img.shape[0]
        batch_range = torch.arange(batch_size)[:, None]
        
        tokens, mask_ids, unmask_ids = self.forward_encoder(img)
        
        decoded_tokens = self.forward_decoder(tokens, mask_ids, unmask_ids)
        
        loss = self.forward_loss(img, decoded_tokens, mask_ids)

        reconstructed = torch.zeros(decoded_tokens.shape, device=device)
        reconstructed[batch_range, unmask_ids] = self.encoder.patch_embed.rearrange(img)[batch_range, unmask_ids]
        reconstructed[batch_range, mask_ids] = decoded_tokens[batch_range, mask_ids]

        return loss, self._patches_to_img(decoded_tokens)
    
    def configure_optimizers(self):
        """Optimization configuration for lightning"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr = self.learning_rate, 
            betas = self.betas, 
            weight_decay = self.weight_decay
        )
        
        scheduler = CosineDecayWarmUp(
            optimizer,
            epochs = self.trainer.max_epochs, 
            warmup_epochs = self.warmup_epochs,
            min_lr = self.min_lr
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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