import torch
import torch.nn as nn
import lightning.pytorch as pl
from .schedulers import CosineDecayWarmUp

class Lightning(pl.LightningModule):
    """A preset autoencoder class for input to a pytorch lightning trainer

    Parameters
    ----------
    autoencoder : nn.Module
        The autoencoder to be trained    
    """
    def __init__(
        self, 
        autoencoder: nn.Module,
        learning_rate: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.01,
        scheduler: str = "plateau",
        factor: float = 0.1,
        patience: int = 30,
        warmup_epochs: int = 40
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['autoencoder'])        
        self.autoencoder = autoencoder
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.scheduler = scheduler  
        self.factor = factor
        self.patience = patience
        self.warmup_epochs = warmup_epochs

    def training_step(self, batch, batch_idx):
        """A single training step for the autoencoder"""
        x, *_ = batch
        loss, _ = self.autoencoder(x)
        self.log("training_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """A single validation step for the autoencoder"""
        x, *_ = batch
        loss, _ = self.autoencoder(x)
        self.log("validation_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """Optimization using Adam with weight decay"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr = self.learning_rate,
            betas = self.betas,
            weight_decay = self.weight_decay                        
        )

        if self.scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor = self.factor,
                patience = self.patience,
                min_lr = self.learning_rate / 50
            )
        elif self.scheduler == 'cosine':
            lr_scheduler = CosineDecayWarmUp(
                optimizer,
                epochs = self.trainer.max_epochs,
                warmup_epochs = self.warmup_epochs,
                min_lr = self.learning_rate / 50
            )
        
        lr_scheduler = {
            'scheduler': lr_scheduler, 
            'name': 'lr_scheduler',
            'monitor': 'validation_loss'
        }

        return [optimizer], [lr_scheduler]
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["validation_loss"])

        if isinstance(sch, CosineDecayWarmUp):
            sch.step()