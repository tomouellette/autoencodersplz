import os
import shutil
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import LinearResidualAE
from autoencodersplz.trainers import AutoencoderTrainer

class TestLinearResidualAE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLinearResidualAE, self).__init__(*args, **kwargs)
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

    def test_train_deterministic(self):
        model = LinearResidualAE(
            img_size = 28,
            in_chans = 1,
            hidden_dim = [64, 64],
            blocks = [2, 2],
            dropout_rate = 0.0,
            with_batch_norm = False,
            latent_dim = 16,
            beta = 0.1,
            kld_weight = None,
            max_temperature = 1000,
            device = None
        )

        trainer = AutoencoderTrainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 10,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_linear_residual_ae_deterministic'
        )
        
        trainer.fit()

    def test_train_stochastic(self):
        model = LinearResidualAE(
            img_size = 28,
            in_chans = 1,
            hidden_dim = [64, 64],
            blocks = [2, 2],
            dropout_rate = 0.0,
            with_batch_norm = False,
            latent_dim = 16,
            beta = 0.1,
            kld_weight = None,
            max_temperature = 1000,
            device = None
        )

        trainer = AutoencoderTrainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 2,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_linear_residual_ae_stochastic'
        )
        
        trainer.fit()

    def tearDown(self):
        # if os.path.exists('tests/train_linear_residual_ae_deterministic'):
        #     shutil.rmtree('tests/train_linear_residual_ae_deterministic')
        
        if os.path.exists('tests/train_linear_residual_ae_stochastic'):
            shutil.rmtree('tests/train_linear_residual_ae_stochastic')

if __name__ == '__main__':
    unittest.main()