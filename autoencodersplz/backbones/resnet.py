import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from ..layers.dimensions import size_conv2d, size_maxpool2d

class ResidualBlock(nn.Module):
    """
    Residual block specifying a shortcut connection with element-wise addition
    
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        change_dim (int): change the dimensionality of convolution to enable addition with input
        stride (int): stride in the first convolution layer and in 1x1 identity convolution if used
    
    Notes:
        | output = F(input, weights_i) + input
    
    References:
        | "Deep Residual Learning for Image Recognition", He et al. 2016
    
    """
    def __init__(self, in_chans: int, out_chans: int, change_dim: bool = False, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chans)
        )
        
        if change_dim:
            self.identity = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride)
        else:
            self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, C, H, W) or (N, C, H/2, W/2) | change_dim
        """
        y = self.convolutions(x)
        x = self.identity(x)
        return F.relu(y + x)

class ResNet(nn.Module):
    """
    A residual neural network with default block structure
    
    Args:
        img_size (tuple): dimensionality of input images
        blocks (list): number of residual blocks in a given residual layer
        channels (list): number of channels for each residual block in a given residual layer
        
    Notes:
        | ResNet-18 can be specified with n_channels = [64, 128, 256, 512] and n_blocks = [2, 2, 2, 2]
    
    References:
        | "Deep Residual Learning for Image Recognition", He et al. 2016
    
    """
    def __init__(
            self, 
            img_size: tuple, 
            channels: Union[list, tuple] = (64, 128, 256, 512), 
            blocks: Union[list, tuple] = (2, 2, 2, 2)
        ):
        super(ResNet, self).__init__()
        self.arguments = locals()
        
        if len(blocks) != len(channels):
            raise ValueError("len(n_blocks) must equal len(channels)")
        
        c, h, w = img_size
        
        # Input convolutions where input channel is usually set to 64
        self.input = nn.Sequential(
            nn.Conv2d(c, channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        c, h, w = (
            channels[0], 
            size_maxpool2d(size_conv2d(h, 7, 2, 3), 3, 2, 1),
            size_maxpool2d(size_conv2d(w, 7, 2, 3), 3, 2, 1)
        )                        
        
        # Build residual layers correcting dimensionality when changing channel size
        residual_blocks = []
        for i, n_channels in enumerate(channels):
            for block in range(blocks[i]):
                if block == 0 and i != 0:
                    residual_blocks.append(
                        ResidualBlock(c, n_channels, change_dim=True, stride=2)
                    )
                    
                    h, w = h//2, w//2
                else:
                    residual_blocks.append(
                        ResidualBlock(c, n_channels)
                    )
                    
                c = n_channels
        
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Store dimensionality information
        self.output_dim = (c, h, w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, n_output)
        """
        x = self.input(x)
        x = self.residual_blocks(x)
        return x
    
class InvertedResidualBlock(nn.Module):
    """
    An inverted residual block for symmetric decoding of latent codes from a ResNet encoder
    
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        change_dim (int): change the dimensionality of convolution to enable addition with input
        stride (int): stride in the first convolution layer and in 1x1 identity convolution if used
        output_padding (int): output padding for the first convolution layer or the identity convolution if used
    
    Notes:
        | output = F(input, weights_i) + input
    
    References:
        | "Deep Residual Learning for Image Recognition", He et al. 2016
    
    """
    def __init__(
            self, 
            in_chans: int, 
            out_chans: int, 
            change_dim: bool = False, 
            stride: int = 1, 
            output_padding: int = 0,
        ):
        super(InvertedResidualBlock, self).__init__()
        self.convolutions = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=3, padding=1, stride=stride, output_padding=output_padding),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.ConvTranspose2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chans)
        )
        
        if change_dim:
            self.identity = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=1, stride=stride, output_padding=output_padding)
        else:
            self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, C, H, W) or (N, C, H/2, W/2) | change_dim
        """
        y = self.convolutions(x)
        x = self.identity(x)
        return F.relu(y + x)

class InvertedResNet(nn.Module):
    """
    An inverted residual neural network for symmetric decoding
    
    Args:
        img_size (tuple): dimensionality of input images
        output_chans (int): number of output channels
        blocks (list): number of residual blocks in a given residual layer
        channels (list): number of channels for each residual block in a given residual layer
        upsample_mode (str): upsampling interpolation mode for rescaling input matrix
        
    Notes:
        | ResNet-18 can be specified with n_channels = [64, 128, 256, 512] and n_blocks = [2, 2, 2, 2]
    
    References:
        | "Deep Residual Learning for Image Recognition", He et al. 2016
    
    """
    def __init__(
            self, 
            img_size: tuple,
            output_chans: int = 3,
            channels: Union[list, tuple] = (64, 128, 256, 512), 
            blocks: Union[list, tuple] = (2, 2, 2, 2),
            upsample_mode: str = 'nearest'
        ):
        super(InvertedResNet, self).__init__()
        self.arguments = locals()
        
        if len(blocks) != len(channels):
            raise ValueError("len(n_blocks) must equal len(channels)")
        
        c, *_ = img_size
        
        # Input convolutions where input channel is usually set to 64
        self.input = nn.Sequential(
            nn.ConvTranspose2d(c, channels[0], kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(channels[0]), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upsample_mode)
        )
        
        c = channels[0]               
        
        # Build residual layers correcting dimensionality when changing channel size
        residual_blocks = []
        for i, n_channels in enumerate(channels):
            for block in range(blocks[i]):
                if block == 0 and i != 0:
                    residual_blocks.append(
                        InvertedResidualBlock(c, n_channels, change_dim=True, stride=2, output_padding=1)
                    )
                else:
                    residual_blocks.append(
                        InvertedResidualBlock(c, n_channels)
                    )
                    
                c = n_channels
        
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.output = nn.Conv2d(channels[-1], output_chans, kernel_size=1, padding=0)            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, n_output)
        """
        x = self.input(x)
        x = self.residual_blocks(x)
        x = self.output(x)
        return x