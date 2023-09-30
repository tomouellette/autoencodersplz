import os
import shutil
import unittest
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from autoencodersplz.models import MAE
from autoencodersplz.trainers import Trainer

class TestMAE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMAE, self).__init__(*args, **kwargs)
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
        model = MAE(
            img_size = 28, 
            in_chans = 1, 
            patch_size = 14,
            embed_dim = 8,
            mlp_ratio = 2,
            depth = 1,
            num_heads = 1,
            mask_ratio = 0.5,
            decoder_embed_dim = 8,
            decoder_depth = 1,
            decoder_num_heads = 1,
        )

        trainer = Trainer(
            model,
            self.train_loader,
            self.test_loader,
            epochs = 2,
            learning_rate = 1e-3,
            save_backbone = True,
            output_dir = 'tests/train_mae'
        )
        
        trainer.fit()

    def tearDown(self):
        if os.path.exists('tests/train_mae'):
            shutil.rmtree('tests/train_mae')        

if __name__ == '__main__':
    unittest.main()