import os
import shutil
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import VQVAE
from autoencodersplz.trainers import Trainer
import lightning.pytorch as pl

class TestVQVAE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestVQVAE, self).__init__(*args, **kwargs)
        self.train_loader = DataLoader(
            MNIST(root='tests/test_data', train=True, download=True, transform=ToTensor()),
            batch_size=32,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            MNIST(root='tests/test_data', train=False, download=True, transform=ToTensor()),
            batch_size=32,
            shuffle=False,
        )

    def test_train(self):
        model = VQVAE(
            img_size = 28,
            in_chans = 1,
            channels = [4],
            blocks = [1],
            codebook_size = 16,
            codebook_dim = 8,
            use_cosine_sim = True,
            kmeans_init = True,
            commitment_weight = 0.5,
            upsample_mode = 'nearest',
            vq_kwargs = {},
        )

        trainer = Trainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 1,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_vqvae'
        )
        
        trainer.fit()

    def test_lightning(self):
        model = VQVAE(
            img_size = 28,
            in_chans = 1,
            channels = [4],
            blocks = [1],
            codebook_size = 16,
            codebook_dim = 8,
            use_cosine_sim = True,
            kmeans_init = True,
            commitment_weight = 0.5,
            upsample_mode = 'nearest',
            vq_kwargs = {},
            learning_rate = 1e-3,
            factor = 0.5,
            patience = 1,
            min_lr = 1e-6,
        )

        trainer = pl.Trainer(max_epochs=1, default_root_dir='tests/train_vqvae_lightning')
        trainer.fit(model, self.train_loader, self.test_loader)

    def tearDown(self):
        if os.path.exists('tests/train_vqvae'):
            shutil.rmtree('tests/train_vqvae')    
        if os.path.exists('tests/train_vqvae_lightning'):
            shutil.rmtree('tests/train_vqvae_lightning')

if __name__ == '__main__':
    unittest.main()