# autoencodersplz

A variety of autoencoder structured models for generative modeling and/or representation learning in pytorch.

## Table of contents

- [Models](#models)  
  - [LinearAE](#linearae)
  - [LinearResidualAE](#linearresidualae)
  - [ConvResidualAE](#convresidualae)
  - [VQVAE](#vqvae)
  - [MAE](#mae)  
- [Training](#training)

## <span id='models'> Models </span>

### <span id='linearae'> LinearAE </span>

A fully-connected autoencoder with a linear/multi-layer perceptron encoder and decoder

```python
from autoencodersplz.models import LinearAE

model = LinearAE(
    img_size = 224,
    in_chans = 3,
    hidden_layers = [64, 64],
    dropout_rate = 0,
    latent_dim = 16,
    beta = 0.1, # beta > 0 = variational
    max_temperature = 1000, # kld temperature annealing
    device = None
)
```

### <span id='linearresidualae'> LinearResidualAE </span>

A fully-connected autoencoder with a linear/multi-layer perceptron residual network backbone and decoder

```python
from autoencodersplz.models import LinearResidualAE

model = LinearResidualAE(
    img_size = 224,
    in_chans = 3,
    hidden_dim = [64, 64],
    blocks = [2, 2],
    dropout_rate = 0.1,
    with_batch_norm = False,
    latent_dim = 16,
    beta = 0.1, # beta > 0 = variational
    max_temperature = 1000, # kld temperature annealing
    device = None
)
```

### <span id='convresidualae'> ConvResidualAE </span>

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

```python
from autoencodersplz.models import ConvResidualAE

model = ConvResidualAE(
    img_size = 224,
    in_chans = 3,
    channels = [64, 128, 256, 512], 
    blocks = [2, 2, 2, 2], 
    latent_dim = 16,
    beta = 0, # beta > 0 = variational
    max_temperature = 1000, # kld temperature annealing
    upsample_mode = 'nearest', # interpolation method
)
```

### <span id='vqvae'> VQVAE </span>

[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

```python
from autoencodersplz.models import VQVAE

model = VQVAE(
    img_size = 224,
    in_chans = 3,
    channels = [64,128,256,512], 
    blocks = [2, 2, 2, 2], 
    codebook_dim = 32,
    code_dim = 64,
    codebook_init = 'uniform',
    metric = 'euclidean',
    beta = 0.25,
    upsample_mode = 'nearest'
)
```
### <span id='mae'> MAE </span>

[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

```python
from autoencodersplz.models import MAE

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

## <span id='training'> Training </span>

<img width="100%" align='center' src='img/training_process.gif'/>

```python
from autoencodersplz.trainers import AutoencoderTrainer

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
