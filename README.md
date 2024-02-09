<p align='center'>    
    <img width="90%" align='center' src="img/logo_caption.png#gh-light-mode-only"/>
    <img width="90%" align='center' src="img/logo_caption_dark.png#gh-dark-mode-only"/>
</p>

---

A variety of autoencoder structured models for generative modeling and/or representation learning in pytorch. Models are mostly designed for usability/extensability/research rather than production implementations. But, go ahead and train some models and reconstruct some things!

## Table of contents

- [Installation](#install)
- [Models](#models)  
  - [LinearAE](#linearae)
  - [LinearResidualAE](#linearresidualae)
  - [ConvResidualAE](#convresidualae)
  - [VQVAE](#vqvae)
  - [FSQVAE](#fsqvae)
  - [MAE](#mae)  
  - [MAEMix](#maemix)
  - [IJEPA](#ijepa)
- [Training](#training)
  - [Basic](#basic-training)
  - [Lightning](#lightning-training)
- [Examples](#examples)
  - [Basic usage](#examples-basic)
- [Future additions](#additions)
- [References](#references)

## <span id='install'> Installation </span>

```bash
pip install autoencodersplz
```

## <span id='models'> Models </span>

### <span id='linearae'> LinearAE </span>

A fully-connected autoencoder with a linear/multi-layer perceptron encoder and decoder

[Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647)

```python
import torch
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

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='linearresidualae'> LinearResidualAE </span>

A fully-connected autoencoder with a linear/multi-layer perceptron residual network encoder and decoder

[Skip Connections Eliminate Singularities](https://arxiv.org/abs/1701.09175)

```python
import torch
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
    device = None,
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='convresidualae'> ConvResidualAE </span>

A convolutional autoencoder with a ResNet encoder and symmetric decoder

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

```python
import torch
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
    device = None,
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='vqvae'> VQVAE </span>

A vector-quantized variational autoencoder with a ResNet encoder and symmetric decoder

[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

```python
import torch
from autoencodersplz.models import VQVAE

model = VQVAE(
    img_size = 224,
    in_chans = 3,
    channels = [64, 128, 256, 512],
    blocks = [2, 2, 2, 2],
    codebook_size = 256,
    codebook_dim = 8,
    use_cosine_sim = True,
    kmeans_init = True,
    commitment_weight = 0.5,
    upsample_mode = 'nearest',
    vq_kwargs = {},
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='fsqvae'> FSQVAE </span>

A finite-scalar quantized variational autoencoder with a ResNet encoder and symmetric decoder

[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)

```python
import torch
from autoencodersplz.models import FSQVAE

model = FSQVAE(
    img_size = 224,
    in_chans = 3,
    channels = [64, 128, 256, 512],
    blocks = [2, 2, 2, 2],
    levels = [8, 6, 5],
    upsample_mode = 'nearest'
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='mae'> MAE </span>

A masked autoencoder with a vision transformer encoder and decoder

[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

```python
import torch
import torch.nn as nn
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
    norm_layer = torch.nn.LayerNorm,
    patch_norm_layer = torch.nn.LayerNorm,
    post_norm_layer = torch.nn.LayerNorm,
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='maemix'> MAEMix </span>

A masked autoencoder with a MLP-mixer encoder and decoder

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

```python
import torch
from autoencodersplz.models import MAEMix

model = MAEMix(
    img_size = 224,
    patch_size = 16,
    in_chans = 3,
    mask_ratio = 0.5,
    embed_dim = 768,
    depth = 12,
    mlp_ratio = 4,
    decoder_embed_dim = 768,
    decoder_depth = 12,
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

### <span id='ijepa'> IJEPA </span>

Image-based joint-embedding predictive architecture (Thanks to <a href='https://github.com/SyouTono242'>Yiran</a> for porting this implementation)

<a href="https://arxiv.org/abs/2301.08243">Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture</a>

```python
import torch
from autoencodersplz.models import IJEPA

model = IJEPA(
    img_size = 224,
    patch_size = 16,
    in_chans = 3,
    embed_dim = 768,
    depth = 12,
    num_heads = 12,
    mlp_ratio = 4,
    embed_dim_predictor = 384,
    predictor_depth = 12,
    num_targets = 4,
    target_aspect_ratio = 0.75,
    target_scale = 0.2,
    context_aspect_ratio = 1.,
    context_scale = 0.9
)

img = torch.rand(1, 3, 224, 224)

loss, reconstructed_img = model(img)
```

## <span id='training'> Training </span>

### <span id='basic-training'> Basic </span>

The `Trainer` class enables basic training using a single CPU or GPU for any model in the `autoencodersplz` library. The `Trainer` class will also automatically save the autoencoder model, backbone/encoder, losses, and a visualization of the training process (`.gif`) if you provide a path to the `output_dir` argument.

```python
from autoencodersplz.trainers import Trainer

trainer = Trainer(
    autoencoder,
    train = train_dataloader,
    valid = valid_dataloader,
    epochs = 128,
    learning_rate = 5e-4,
    betas = (0.9, 0.95),
    weight_decay = 0.05,
    patience = 10,
    scheduler = 'plateau',
    save_backbone = True,
    show_plots = False,
    output_dir = 'training_run/',
    device = None,
)

trainer.fit()
```

By default, `Trainer` uses an `AdamW` optimizer and either a `CosineDecay` ('cosine') or `ReduceLROnPlateau` ('plateau') scheduler. If you want to use different optimizers or schedulers, just re-assign a new optimizer or scheduler to the `.optimizer` or `.scheduler` attributes (with `trainer.model.parameters()`) prior to calling `trainer.fit()`.

### <span id='lightning-training'> Lightning </span>

To make it easier to scale to multi-gpu/distributed training, all `autoencodersplz` models are configured for use with [pytorch lightning](https://lightning.ai/docs/pytorch). Each model is setup with a default optimizer and scheduler and can be directly called by the pytorch lightning trainer. See an example below.

```python
import lightning.pytorch as pl
from autoencodersplz.models import FSQVAE

model = FSQVAE(
    img_size = 28,
    in_chans = 1,
    channels = [8, 16],
    blocks = [1, 1],
    levels = [8],
    upsample_mode = 'nearest'
    learning_rate = 1e-3,
    factor = 0.1,
    patience = 30,
    min_lr = 1e-6
) 

trainer = pl.Trainer(gpus=4, max_epochs=256)

trainer.fit(model, train_dataloader, valid_dataloader)
```

## <span id='examples'> Examples </span>

### <span id='examples-basic'> Basic usage </span>

Here's a basic example of training a fully connected autoencoder on MNIST. The data is downloaded and loaded and then the autoencoder is fit. The training info is logged to the output directory (`training/`) and a GIF of the training routine is generated for visual inspection.

<img width="100%" align='center' src='img/training_process.gif'/>

```python
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from autoencodersplz.models import LinearAE
from autoencodersplz.trainers import Trainer

train_loader = DataLoader(
    MNIST(root='data/', train=True, download=True, transform=ToTensor()),
    batch_size = 32,
    shuffle = True,
)

test_loader = DataLoader(
    MNIST(root='data/', train=False, download=True, transform=ToTensor()),
    batch_size = 32,
    shuffle = False,
)

model = LinearAE(
    img_size = 28,
    in_chans = 1,
    hidden_layers = [256, 128],
    dropout_rate = 0,
    latent_dim = 32,
    beta = 0,
)

trainer = Trainer(
    model, 
    train_loader, 
    test_loader, 
    epochs = 32, 
    learning_rate = 1e-3, 
    output_dir = 'training/'
)
        
trainer.fit()
```

## <span id='additions'> Future additions </span>

- [ ] [Evolved Part Masking for Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Evolved_Part_Masking_for_Self-Supervised_Learning_CVPR_2023_paper.pdf)
- [ ] [Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Reconstruction_Autoencoder-Based_Out-of-Distribution_Detection_CVPR_2022_paper.pdf)
- [ ] [Catch Missing Details: Image Reconstruction with Frequency Augmented
Variational Autoencoder](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Catch_Missing_Details_Image_Reconstruction_With_Frequency_Augmented_Variational_Autoencoder_CVPR_2023_paper.pdf)
- [ ] [Dual Contradistinctive Generative Autoencoder](https://openaccess.thecvf.com/content/CVPR2021/papers/Parmar_Dual_Contradistinctive_Generative_Autoencoder_CVPR_2021_paper.pdf)
- [ ] [Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial
Representation Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf)
- [ ] [Guided Variational Autoencoder for Disentanglement Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Guided_Variational_Autoencoder_for_Disentanglement_Learning_CVPR_2020_paper.pdf)
- [ ] [MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_MAGE_MAsked_Generative_Encoder_To_Unify_Representation_Learning_and_Image_CVPR_2023_paper.pdf)

## References

```bibtex
@article{hinton2006reducing,
  title = {Reducing the dimensionality of data with neural networks},
  author = {Geoffrey Hinton and Ruslan Salakhutdinov},
  url = {10.1126/science.1127647},  
  year = {2006},
}
```

```bibtex
@article{orhan2018skip,
    title = {Skip Connections Eliminate Singularities},
    author = {Emin Orhan and Xaq Pitkow},
    url = {https://arxiv.org/abs/1701.09175},
    year = {2018},    
}
```

```bibtex
@article{he2015deep,
    title = {Deep Residual Learning for Image Recognition}, 
    author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    url = {https://arxiv.org/abs/1512.03385},
    year = {2016},
}
```
```bibtex
@misc{oord2018neural,
    title={Neural Discrete Representation Learning}, 
    author={Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
    url = {https://arxiv.org/abs/1711.00937},
    year={2017},
}
```

```bibtex
@misc{mentzer2023finite,
    title = {Finite Scalar Quantization: VQ-VAE Made Simple}, 
    author = {Fabian Mentzer and David Minnen and Eirikur Agustsson and Michael Tschannen},    
    url = {https://arxiv.org/abs/2309.15505},
    year = {2023},
}
```

```bibtex
@misc{he2021masked,
    title = {Masked Autoencoders Are Scalable Vision Learners}, 
    author = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollár and Ross Girshick},
    url = {https://arxiv.org/abs/2111.06377},
    year = {2021},
}
```

```bibtex
@misc{tolstikhin2021mlpmixer,
    title = {MLP-Mixer: An all-MLP Architecture for Vision}, 
    author = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Andreas Steiner and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    url = {https://arxiv.org/abs/2105.01601},
    year = {2021},
}
```

```bibtex
@misc{assran2023selfsupervised,
    title = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture}, 
    author = {Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
    url = {https://arxiv.org/abs/2301.08243},
    year = {2023},
}
```
