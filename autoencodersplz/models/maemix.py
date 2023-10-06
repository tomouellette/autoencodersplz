import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from lightning import LightningModule
from typing import Union, Tuple

from ..backbones.mlp_mixer import MLPMixer
from ..layers.dimensions import to_tuple, collect_batch
from ..trainers.schedulers import CosineDecayWarmUp

class MAEMix(LightningModule):
    """A masked autoencoder with a MLP-mixer backbone/encoder and decoder

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
    mlp_ratio : float, optional
        The ratio of the hidden dimension to the embedding dimension, by default 4
    decoder_embed_dim : int, optional
        The dimension of the decoder embedding, by default 768
    decoder_depth : int, optional
        The number of decoder transformer blocks, by default 12
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
        mlp_ratio: float = 4.,
        decoder_embed_dim: int = 768,
        decoder_depth: int = 12,
        learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        min_lr: float = 1e-6,
        warmup_epochs: int = 40,
    ):
        super(MAEMix, self).__init__()
        self.arguments = locals()
        
        self.mask_ratio = mask_ratio
        self.decoder_embed_dim = decoder_embed_dim

        self.encoder = MLPMixer(
            img_size = img_size,
            in_chans = in_chans,
            patch_size = patch_size,
            dim = embed_dim,
            depth = depth,             
            expansion_factor = mlp_ratio,
            expansion_factor_token = 0.5, 
            dropout = 0.,
            num_classes = 100,
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)

        num_patches = self.encoder.num_patches
        self.patch_height, self.patch_width  = to_tuple(patch_size)

        # decoder x'|z
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(decoder_embed_dim))
        
        self.decoder = MLPMixer(
            img_size = img_size,
            in_chans = in_chans,
            patch_size = patch_size,
            dim = decoder_embed_dim,
            depth = decoder_depth,             
            expansion_factor = mlp_ratio,
            expansion_factor_token = 0.5, 
            dropout = 0.,
            num_classes = 100,
        )

        self.decoder_blocks = self.decoder.forward_mix
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
        tokens = self.encoder.forward_embed(img)

        # transformer blocks; masking done after encoding w/ MLP-mixer 
        # although we could refactor the mixing layers to have masking first
        tokens = self.encoder.forward_mix(tokens)

        # masking: (n, num_patches, embed_dim) -> (n, num_patches * (1-mask_ratio), embed_dim)
        tokens, mask_ids, unmask_ids = self.random_masking(tokens, self.mask_ratio)

        tokens = self.encoder_norm(tokens)

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
        
        # store unmasked tokens
        unmasked_tokens = tokens + self.decoder_pos_embed(unmask_ids)

        # expand mask tokens to batch
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
        
        img = self.encoder.rearrange(img)[batch_range, mask_ids]
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
        reconstructed[batch_range, unmask_ids] = self.encoder.rearrange(img)[batch_range, unmask_ids]
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
        self.log("train_loss", loss, batch_size=batch.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for lightning"""
        batch = collect_batch(batch)
        loss, _ = self.forward(batch)
        self.log("val_loss", loss, batch_size=batch.size(0))