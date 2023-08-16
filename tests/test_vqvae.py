import os
import shutil
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import VQVAE
from autoencodersplz.trainers import AutoencoderTrainer

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
            channels = [16],
            blocks = [2],
            codebook_dim = 256,
            code_dim = 64,
            beta = 0.25,            
            upsample_mode = 'nearest',
            device = None
        )

        trainer = AutoencoderTrainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 2,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_vqvae'
        )
        
        trainer.fit()

    def tearDown(self):
        if os.path.exists('tests/train_vqvae'):
            shutil.rmtree('tests/train_vqvae')    

if __name__ == '__main__':
    unittest.main()