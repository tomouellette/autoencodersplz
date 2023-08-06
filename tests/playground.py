from autoencodersplz.models.mae import MAE
from autoencodersplz.trainers import AutoencoderTrainer

# MNIST train and test loaders
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_loader = DataLoader(
    MNIST(root='data', train=True, download=True, transform=ToTensor()),
    batch_size=32,
    shuffle=True,
)

test_loader = DataLoader(
    MNIST(root='data', train=False, download=True, transform=ToTensor()),
    batch_size=32,
    shuffle=False,
)

model = MAE(
    img_size=28, 
    in_chans=1, 
    patch_size=4,
    embed_dim=32,
    mlp_ratio=2,
    depth=2,
    num_heads=2,
    mask_ratio = 0.7,
    decoder_embed_dim = 32,
    decoder_depth = 2,
    decoder_num_heads = 2,
)

from autoencodersplz.models import ResAE

model = ResAE(
    img_size = 28,
    in_chans = 1,
    channels = [64], 
    blocks = [2], 
    latent_dim = 16,
    beta = 0.1,
    kld_weight = None,
    max_temperature = 1000,
    upsample_mode = 'nearest',
    device = None
)

trainer = AutoencoderTrainer(
    model,
    train_loader,
    test_loader,
    epochs=2,
    learning_rate=1e-3,
    save_backbone=True,
    output_dir='tests/test_runs_2'
)

trainer.fit()

