import os
import shutil
import unittest

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import IJEPA
from autoencodersplz.trainers import Trainer

class TestIJEPA(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestIJEPA, self).__init__(*args, **kwargs)
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
        model = IJEPA(
            img_size = 28,
            patch_size = 4,
            in_chans = 1,
            embed_dim = 32,
            depth = 2,
            num_heads = 2,
            embed_dim_predictor = 16,
            predictor_depth = 12,
            num_targets = 4,
            target_aspect_ratio = 0.75,
            target_scale = 0.2,
            context_aspect_ratio = 1.,
            context_scale = 0.9
        )
        
        trainer = Trainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 1,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_ijepa'
        )
        
        trainer.fit()        
    
    def tearDown(self):
        if os.path.exists('tests/train_ijepa'):
            shutil.rmtree('tests/train_ijepa')
            
if __name__ == '__main__':
    unittest.main()