import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A multi-layer (fully-connected) perceptron
    
    Args:
        input_dim (int): dimension of input features
        hidden_layers (list): list of integers specifying the number of units in each hidden layer
        output_dim (int): dimension of output features
        dropout_rate (float): dropout rate
        activation (torch.nn.modules.activation): pytorch activation function
    
    """
    def __init__(
            self, 
            input_dim: int, 
            hidden_layers: list, 
            output_dim: int, 
            dropout_rate: float = 0, 
            activation = nn.ReLU
        ):
        super(MLP, self).__init__()
        self.arguments = locals()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if (len(hidden_layers) == 0) | (type(hidden_layers) != list):
            raise ValueError("Hidden layers must be a list of integers.")
        
        if not hasattr(torch.nn.modules.activation, str(activation()).split('(')[0]):
            raise ValueError("Activation must be a torch.nn.modules.activation function.")
        
        if (dropout_rate < 0) | (dropout_rate > 1):
            raise ValueError("Dropout must be between 0 and 1.")
        
        layers = [nn.Linear(input_dim, hidden_layers[0]), activation()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(activation())
        
        layers.append(nn.Linear(hidden_layers[-1], output_dim)) 
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device        
        return self.layers(x)

class LinearResidualBlock(nn.Module):
    """
    A residual block for a multi-layer perceptron

    Args:
        input_dim (int): input dimension
        dropout_rate (float): dropout rate
        with_batch_norm (bool): whether to use batch normalization
        zero_initialization (bool): whether to initialize weights and biases to zero
    
    """
    def __init__(
            self,
            input_dim: int,
            dropout_rate = 0.0,
            with_batch_norm: bool = False,
            zero_initialization: bool = False,
        ):
        super().__init__()        
        layers = []
        for _ in range(2):
            if with_batch_norm:
                layers.append(nn.BatchNorm1d(input_dim, eps=1e-3))

            layers.append(nn.Linear(input_dim, input_dim))                    
            layers.append(nn.ReLU())

            if zero_initialization:
                nn.init.uniform_(layers[-2].weight, -1e-3, 1e-3)
                nn.init.uniform_(layers[-2].bias, -1e-3, 1e-3)
            
        layers.append(nn.Dropout(p=dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return y + x


class LinearResidualNet(nn.Module):
    """
    A multi-layer perceptron with skip connections

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        hidden_dim (list): list of hidden layer dimensions
        blocks (list): list of number of residual blocks per hidden layer
        dropout_rate (float): dropout rate
        with_batch_norm (bool): whether to use batch normalization in residual blocks
        zero_initialization (bool): whether to initialize weights and biases to zero
    
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: list = [64, 64],
            blocks: int = [2, 2],
            dropout_rate: float = 0.0,
            with_batch_norm: bool = False,
            zero_initialization: bool = False,
        ):        
        super(LinearResidualNet, self).__init__()
        self.arguments = locals()
        assert len(blocks) == len(hidden_dim), "n_blocks must equal len(hidden_dim)"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = [nn.Linear(input_dim, hidden_dim[0])]
        for i, hidden in enumerate(hidden_dim):
            for _ in range(blocks[i]):
                layers.append(LinearResidualBlock(
                    input_dim = hidden, 
                    dropout_rate = dropout_rate,
                    with_batch_norm = with_batch_norm,
                    zero_initialization = zero_initialization,
                ))

            if i != len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden, hidden_dim[i+1]))

        layers.append(nn.Linear(hidden_dim[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.device = x.device
        return self.layers(x)