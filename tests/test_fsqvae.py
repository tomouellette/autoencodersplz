import os
import shutil
import torch
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import FSQVAE
from autoencodersplz.trainers import Trainer

class TestFSQVAE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFSQVAE, self).__init__(*args, **kwargs)
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
        model = FSQVAE(
            img_size = 28,
            in_chans = 1,
            channels = [4],
            blocks = [1],
            levels = [4],
            upsample_mode = 'nearest'
        )

        trainer = Trainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 1,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_fsqvae'
        )
        
        trainer.fit()

    def test_input_sizes(self):
        """
        Test that the model can handle inputs of different sizes
        """
        for d in [64, 128, 256, 512]:
            for levels in [[4], [8,6,4], [6,5,4,3], [5,3,2,1]]:
                x = torch.rand(1, 3, d, d)
                model = FSQVAE(
                    img_size = d,
                    in_chans = 3,
                    channels = [d//4, d//2],
                    blocks = [1, 1],
                    levels = levels,
                    upsample_mode = 'nearest'
                )
                
                _ = model(x)
        
    def tearDown(self):
        if os.path.exists('tests/train_fsqvae'):
            shutil.rmtree('tests/train_fsqvae')    

if __name__ == '__main__':
    unittest.main()