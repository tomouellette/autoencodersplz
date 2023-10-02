import os
import shutil
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from autoencodersplz.models import LinearAE
from autoencodersplz.trainers import Lightning

class TestLightning(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLightning, self).__init__(*args, **kwargs)
        self.train_loader = DataLoader(
            MNIST(root='tests/test_data', train=True, download=True, transform=ToTensor()),
            batch_size=32,
            num_workers=4,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            MNIST(root='tests/test_data', train=False, download=True, transform=ToTensor()),
            batch_size=32,
            num_workers=4,
            shuffle=False,
        )

    def test_lightning_training_plateau(self):
        """
        Test training using a lightning module wrapper
        """
        autoencoder = LinearAE(
            img_size = 28,
            in_chans = 1,
            hidden_layers = [4, 4],
            dropout_rate = 0,
            latent_dim = 16,
            beta = 0.,
            kld_weight = None,
            max_temperature = 1000,
        )        

        model = Lightning(
            autoencoder = autoencoder,
            learning_rate = 1e-3,
            betas = (0.9, 0.999),
            weight_decay = 0.01,
            scheduler = "plateau",
            factor = 0.1,
            patience = 30,
        ) 

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(max_epochs=1, callbacks=[lr_monitor], default_root_dir='tests/')
        trainer.fit(model, self.train_loader, self.test_loader)

    def test_lightning_training_cosine(self):
        """
        Test training using a lightning module wrapper
        """
        autoencoder = LinearAE(
            img_size = 28,
            in_chans = 1,
            hidden_layers = [4, 4],
            dropout_rate = 0,
            latent_dim = 16,
            beta = 0.,
            kld_weight = None,
            max_temperature = 1000,
        )        

        model = Lightning(
            autoencoder = autoencoder,
            learning_rate = 1e-3,
            betas = (0.9, 0.999),
            weight_decay = 0.01,
            scheduler = "cosine",
            warmup_epochs = 1,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(max_epochs=1, callbacks=[lr_monitor], default_root_dir='tests/')
        trainer.fit(model, self.train_loader, self.test_loader)

    def tearDown(self) -> None:
        """
        Remove the test directory
        """
        if os.path.exists('tests/lightning_logs/'):
            shutil.rmtree('tests/lightning_logs/')

if __name__ == '__main__':
    unittest.main()