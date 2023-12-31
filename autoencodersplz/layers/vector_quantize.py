import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VectorQuantizer(nn.Module):
    """A vector quantization layer for discretizing continuous latent vectors into discrete codes

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vectors (e.g. output of a convolutional encoder), defaults to 256
    codebook_dim : int
        Number of vectors in the codebook, defaults to 512
    code_dim : int
        Dimensionality of the vectors in the codebook, defaults to 64
    codebook_init : str
        Initialization mode for the codebook vectors, defaults to 'uniform'
    metric : str
        Distance metric to use for finding nearest neighbours in the codebook, defaults to 'euclidean'
    beta : float
        Weighting of the embedding loss term (commitment cost), defaults to 0.25

    Notes
    -----
    Replaced this basic VQ implementation with lucidrains vector-quantize-pytorch implementation
    throughout the codebase. This implementation is kept for reference.
    """

    def __init__(
        self,
        latent_dim: int,
        codebook_dim: int,
        code_dim: int,
        codebook_init: str = 'uniform',
        metric: str = 'euclidean',
        beta: float = 0.25
    ):
        super(VectorQuantizer, self).__init__()
        self.arguments = locals()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.codebook_dim = codebook_dim
        self.code_dim = code_dim
        self.codebook_init = codebook_init
        self.metric = metric
        self.beta = beta
        
        self.project_in = nn.Conv2d(latent_dim, code_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_dim, code_dim)
        self.project_out = nn.Conv2d(code_dim, latent_dim, kernel_size=1)
    
    def _codebook_init(self, mode: str) -> None:
        """
        Initialize the values within the codebook

        Parameters
        ----------
        mode : str
            Initialization mode for the codebook vectors        
        """
        if mode == 'uniform':
            self.codebook.weight.data.uniform_(-1/self.codebook_dim, 1/self.codebook_dim).to(self.device)
    
    def _find_codebook_neighbours(self, z: torch.Tensor) -> torch.Tensor:
        """Find index of nearest neighbour in codebook for each latent vector"""
        if self.metric == 'euclidean':
            distances = (
                z.pow(2).sum(1).unsqueeze(1)
                - 2 * z @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(1)
            )
        
        indices = distances.argmin(1)
        
        return indices
    
    def forward_encoder(self, z: torch.Tensor) -> torch.Tensor:
        """Generate discrete codes given continuous latent inputs"""
        b, c, h, w = z.shape
        
        # Project from input latent vectors
        z = self.project_in(z)
        
        # Re-arrange to independetly quantize each dimension
        z_e = rearrange(z, 'b c h w -> (b h w) c')
        
        # Find closest neighbour in the codebook
        indices = self._find_codebook_neighbours(z_e).view(b, h, w)
        
        return z, indices
    
    def forward_decoder(self, indices: torch.Tensor) -> torch.Tensor:
        """Generate quantized vectors from codebook indices"""
        # Quantization by decoding indices using the corresponding codebook vectors
        z_q = F.embedding(indices, self.codebook.weight)       
        
        # Re-arrange decoded vectors match input latent vector shape
        z_q = rearrange(z_q, 'b h w c -> b c h w')        
        
        return z_q
    
    def forward_loss(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        """Calculate loss for training the input latent vectors and quantized latent vectors"""
        codebook_loss = F.mse_loss(z_q, z.detach())
        embedding_loss = F.mse_loss(z_q.detach(), z)       
        loss = codebook_loss + self.beta * embedding_loss
        return loss
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.device = z.device

        if self.codebook_init:
            self._codebook_init(self.codebook_init)
            self.codebook_init = None
        
        # Quantize
        z, indices = self.forward_encoder(z)
        z_q = self.forward_decoder(indices)

        # Codebook and embedding loss
        loss = self.forward_loss(z, z_q)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Project back to latent dimensionality
        z_q = self.project_out(z_q)

        return loss, z_q