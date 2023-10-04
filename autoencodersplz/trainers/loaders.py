import torch.nn as nn
import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader

class WebDatasetLoader:
    """Specify web dataset streams for training deep learning models

    Parameters
    ----------
    path : str
        Path (local directory or path) to web dataset formatted tar file
    transformations : list, optional
        List of ordered torchvision transforms to apply to the images, by default None
    n_test : int, optional
        Number of test samples to hold out, by default 0
    shuffle_size : int, optional
        Number of samples to load into memory for shuffling, by default 64
    batch_size : int, optional
        Number of samples per batch, by default 32
    decoding : str, optional
        Color model (e.g. rgb) for decoding images in web dataset, by default 'rgb'
    image_type : str, optional
        Image type for images in web dataset, by default 'png'
    label_type : str, optional
        Label type for labels in web dataset, by default 'json'
    length : int, optional
        Number of samples in the dataset, by default None    
    """
    def __init__(
        self,
        path: str,
        transformations: list = None,
        n_test: int = 0,
        shuffle_size: int = 64,
        batch_size: int = 32,
        decoding: str = 'rgb',
        image_type: str = 'png',
        label_type: str = 'json',
        length: int = None,
    ):
        if isinstance(transformations, type(None)):
            self.transformations = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transformations = transforms.Compose([
                transforms.ToTensor(), *transformations
            ])
        
        # Store configuration
        self.path = path
        self.n_test = n_test
        self.shuffle_size = shuffle_size
        self.decoding = decoding
        self.image_type = image_type
        self.label_type = label_type
        self.batch_size = batch_size
        
        # Store the number of samples
        if isinstance(length, int):
            self.length = length
        else:
            self.length = 0
            for _ in wds.WebDataset(self.path): self.length += 1
    
    def dataloader(self, **kwargs) -> DataLoader:
        """Returns a dataloader for the entire dataset

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to wds.WebLoader
        
        Returns
        -------
        wds.WebLoader
            Dataloader for entire dataset
        
        Notes
        -----
        length = math.ceil(self.length / self.batch_size)
        """
        dataset = (
            wds.WebDataset(self.path)
            .shuffle(self.shuffle_size)
            .decode(self.decoding)
            .to_tuple(self.image_type, self.label_type)
            .map_tuple(self.transformations, nn.Identity())
        )

        return DataLoader(dataset.batched(self.batch_size), batch_size=None, **kwargs)
            
    def train_dataloader(self, **kwargs) -> DataLoader:
        """Returns a dataloader for the training dataset with len(dataset)-n_test samples

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to wds.WebLoader
        
        Returns
        -------
        wds.WebLoader
            Dataloader for training dataset
        
        Notes
        -----
        length = math.ceil((self.length - self.n_test) / self.batch_size)
        """
        dataset = (
            wds.WebDataset(self.path)
            .slice(0, self.length - self.n_test, 1)
            .shuffle(self.shuffle_size)
            .decode(self.decoding)
            .to_tuple(self.image_type, self.label_type)
            .map_tuple(self.transformations, nn.Identity())
        )
        
        return DataLoader(dataset.batched(self.batch_size), batch_size=None, **kwargs)
    
    def test_dataloader(self, **kwargs) -> DataLoader:
        """Returns a dataloader for the test dataset with n_test samples

        Parameters
        ----------
        kwargs
            Keyword arguments to pass to wds.WebLoader
        
        Returns
        -------
        wds.WebLoader
            Dataloader for test/validation dataset
        
        Notes
        -----
        length = math.ceil(self.n_test / self.batch_size)
        """
        dataset = (
            wds.WebDataset(self.path)
            .slice(self.length - self.n_test, self.length, 1)
            .decode(self.decoding)
            .to_tuple(self.image_type, self.label_type)
            .map_tuple(transforms.ToTensor())
        )
        
        return DataLoader(dataset.batched(self.batch_size), batch_size=None, **kwargs)