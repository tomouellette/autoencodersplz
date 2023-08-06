import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.dimensions import size_conv2d, size_maxpool2d

class ResidualBlock(nn.Module):
    """
    Residual block specifying a shortcut connection with element-wise addition
    
    Args:
        n_channels (int): number of channels in each convolution layer
        change_channels (int): change the number of channels before adding identity
        stride (int): stride in the first convolution layer and in 1x1 identity convolution if used
    
    Notes:
        | output = F(input, weights_i) + input
    
    References:
        | "Deep Residual Learning for Image Recognition", He et al. 2016
    
    """
    def __init__(self, n_channels: int, change_channels: bool = False, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.convolutions = nn.Sequential(
            nn.LazyConv2d(n_channels, kernel_size=3, padding=1, stride=stride),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(n_channels, kernel_size=3, padding=1, stride=1),
            nn.LazyBatchNorm2d()
        )
        
        if change_channels:
            self.identity = nn.LazyConv2d(n_channels, kernel_size=1, stride=stride)
        else:
            self.identity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        I/O: (N, C, H, W) -> (N, C, H, W) or (N, C, H/2, W/2) | change_channels
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
            channels: list = [64, 128, 256, 512], 
            blocks: list = [2, 2, 2, 2]
        ):
        super(ResNet, self).__init__()
        self.arguments = locals()
        
        if len(blocks) != len(channels):
            raise ValueError("len(n_blocks) must equal len(channels)")
        
        c, h, w = img_size
        
        # Input convolutions where input channel is usually set to 64
        self.input = nn.Sequential(
            nn.LazyConv2d(channels[0], kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), 
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
                        ResidualBlock(n_channels, change_channels=True, stride=2)
                    )
                    
                    h, w = h//2, w//2
                else:
                    residual_blocks.append(
                        ResidualBlock(n_channels)
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