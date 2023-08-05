# autoencodersplz

A variety of autoencoder structued models for generative modeling and/or representation learning in pytorch.

## Table of contents

## Models

### MAE

```python
from tmim.models import MAE

model = MAE(
    img_size = 224,
    patch_size = 16,
    in_chans = 3,
    mask_ratio = 0.5,
    embed_dim = 768,
    depth = 12,
    num_heads = 12,
    mlp_ratio = 4,
    pre_norm = False,
    decoder_embed_dim = 768,
    decoder_depth = 12,
    decoder_num_heads = 12,
    norm_layer = nn.LayerNorm,
    patch_norm_layer = nn.LayerNorm,
    post_norm_layer = nn.LayerNorm,
)
```

## Training

<img width="100%" align='center' src='img/training_process.gif'/>

```python
from tmim.trainers import AutoencoderTrainer

trainer = AutoencoderTrainer(
    model,
    train = train_dataloader,
    valid = valid_dataloader,
    epochs = 128,
    learning_rate = 5e-4,
    betas = (0.9, 0.95),
    patience = 10,
    scheduler = 'plateau',
    save_backbone = False,
    show_plots = False,
    output_dir = 'training_run/',
    device = None,
)

trainer.fit()
```

By default, `AutoencoderTrainer` uses an `AdamW` optimizer and either a `CosineDecay` ('cosine') or `ReduceLROnPlateau` ('plateau') scheduler. If you want to use different optimizers or schedulers, just re-assign a new optimizer or scheduler to the `.optimizer` or `.scheduler` attributes (with `trainer.model.parameters()`) prior to calling `trainer.fit()`.
