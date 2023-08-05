import os
import math
import torch
import shutil
import imageio
import numpy as np
import seaborn as sns
import torch.nn as nn
from typing import IO
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .schedulers import CosineDecayWarmUp
from .loggers import AutoencoderLogger

"""
Classes for fitting self-supervised related models
"""

class AutoencoderTrainer:
    """
    Trainer class for fitting autoencoder-based representation models
    
    Args:
        autoencoder (nn.Module): autoencoder model to fit
        train (DataLoader): training data
        valid (DataLoader): validation data
        epochs (int): maximum number of training epochs
        learning_rate (float): learning rate for optimizer
        patience (int): number of epochs to wait before reducing learning rate
        show_plots (bool): if True, then show plots at end of each epoch
        device (str): device to use for training; if None, then automatically detect GPU or CPU
    
    """
    def __init__(
            self, 
            autoencoder: nn.Module,
            train: DataLoader,
            valid: DataLoader,
            epochs: int = 128,
            warmup_epochs: int = 10,
            learning_rate: float = 1.5e-4,
            betas: float = (0.9, 0.95),
            patience: int = 10,
            scheduler: str = 'plateau',
            save_backbone: bool = False,
            show_plots: bool = False,
            output_dir: str = None,
            device: str = None,
            **kwargs
        ) -> None:
        if isinstance(device, type(None)):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model, self.train, self.valid = autoencoder.to(self.device), train, valid
        
        # store hyperparameters
        self.epochs = epochs        
        self.learning_rate = learning_rate
        self.patience = patience
        self.show_plots = show_plots
        self.output_dir = output_dir
        self.save_backbone = save_backbone
        
        # instantiate logger, optimizer, scheduler, and callbacks
        self.logger = AutoencoderLogger()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.05, betas=betas)

        if scheduler == 'cosine':
            self.scheduler = CosineDecayWarmUp(self.optimizer, epochs=epochs, warmup_epochs=warmup_epochs, min_lr=learning_rate*0.1)
        elif scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.1)
        else:
            raise ValueError("scheduler must be one of 'cosine' or 'plateau'")
        
        if not isinstance(output_dir, type(None)):
            if not os.path.exists(os.path.join(self.output_dir)):
                os.makedirs(os.path.join(self.output_dir), exist_ok=True)
        
    def training_step(self, x: torch.Tensor) -> dict:        
        """
        Training step for a single batch of reconstruction and pseudo-label prediction
        """
        loss, xhat = self.model(x)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), xhat
    
    @torch.no_grad()
    def validation_step(self, x: torch.Tensor) -> dict:
        """
        Validation step for a single batch of reconstruction and pseudo-label prediction
        """
        loss, xhat = self.model(x)            
        return loss.item(), xhat
    
    @torch.no_grad()
    def generate_plots(self, x: torch.Tensor, xhat: torch.Tensor, epoch: int, losses: list) -> IO:
        """
        Plot a set of reconstructions and training progress end of epoch
        
        Args:
            batch (tuple): a single batch
            xhat (torch.Tensor): reconstructed images for all samples in validation set
            labels (torch.Tensor): if self.with_labels is True, then ground-truth labels for each sample
        
        """        
        # generate reconstructions and set shape to be compatible with matplotlib        
        x = x.detach().cpu().permute(0,2,3,1)
        xhat = xhat.detach().cpu().permute(0,2,3,1)

        # instantiate figure
        fig, ax = plt.subplot_mosaic([
            [0,0,1,1,2,2,3,3,20,20],
            [4,4,5,5,6,6,7,7,40,40]
        ], figsize=(7,3))
        fig.set_facecolor('black')

        # plot reconstructions
        ax[0].set_title('Images $x$', loc='left', c='w', fontsize=10)
        ax[4].set_title('Reconstructed $x|z$', loc='left', c='w', fontsize=10)
        for i in range(4):
            # images
            ax[i].imshow(
                np.clip(x[i] / x[i].max(), 0, 1), 
                cmap=None if x.shape[-1] != 1 else 'gray'
            )
            # reconstructed
            ax[4+i].imshow(
                np.clip(xhat[i] / xhat[i].max(), 0, 1),
                cmap=None if x.shape[-1] != 1 else 'gray'
            )
            
            ax[i].axis('off')
            ax[4+i].axis('off')
            ax[i].set_facecolor('black')
            ax[4+i].set_facecolor('black')
        
        # plot loss curve
        ax[20].plot(self.logger.log['epoch'], self.logger.log['loss'], c='#ff99c8')
        ax[20].axhline(self.logger.log['loss'][0], c='w', alpha=0.5, linestyle='dashed')
        
        ax[20].set_xlim(-1, self.epochs)
        ax[20].set_ylim(
            np.min(self.logger.log['loss']) - np.abs(np.max(self.logger.log['loss']))*0.05, 
            np.max(self.logger.log['loss']) + np.abs(np.max(self.logger.log['loss']))*0.05
        )
        ax[20].set_facecolor('black')
        ax[20].set_xticks([])
        ax[20].set_xticklabels([])
        ax[20].set_yticks([])
        ax[20].set_yticklabels([])
        ax[20].set_ylabel('Loss', fontsize=10, c='w')
        
        for pos in ['right', 'top']: 
            ax[20].spines[pos].set_visible(False)            

        for pos in ['left', 'bottom']:
            ax[20].spines[pos].set_color('w')
        
        from scipy.stats import gaussian_kde

        # plot delta loss
        ax[40].axvline(x=0, c='w', alpha=0.5, linestyle='dashed')        
        if epoch > 0:
            delta_losses = np.array(losses[1]) - np.array(losses[0])

            kde = gaussian_kde(delta_losses)
            
            g = sns.kdeplot(x=delta_losses, c='w', ax=ax[40], linewidth=0.1, label=f'{epoch+1}')
            xmin, xmax = g.get_xlim()
            xd = np.linspace(xmin, xmax, 1000)
            yd = kde(xd)

            x_left, y_left, x_right, y_right = xd[xd < 0], yd[xd < 0], xd[xd >= 0], yd[xd >= 0]
            g.fill_between(x=x_left, y1=y_left, color='#fcf6bd', alpha=.5)
            g.fill_between(x=x_right, y1=y_right, color='#d0f4de', alpha=.5)
        
        ax[40].set_facecolor('black')
        ax[40].get_yaxis().set_visible(False)
        ax[40].set_xlabel('$\Delta$ Loss', fontsize=10, c='w')
        ax[40].set_xticks([])
        ax[40].set_xticklabels([])
        ax[40].spines['bottom'].set_color('w')

        for pos in ['right', 'top', 'left']: 
            ax[40].spines[pos].set_visible(False)

        plt.suptitle(f'Epoch {epoch+1}', x=0.025, y=0.925, ha='left', c='w', fontsize=10)
        fig.tight_layout()
        
        if (self.show_plots) & (isinstance(self.output_dir, type(None))):
            plt.show()
        elif isinstance(self.output_dir, type(None)):
            pass
        else:
            if not os.path.exists(os.path.join(self.output_dir, 'training_plots')):
                os.makedirs(os.path.join(self.output_dir, 'training_plots'))    
            
            plt.savefig(os.path.join(self.output_dir, 'training_plots/', f'epoch_{epoch+1}.png'), dpi=100)
            plt.close()
    
    def generate_gif(self) -> IO: 
        """
        Converts all training plots into a single gif

        Returns:
            IO: saves a gif of training process to self.output_dir
        
        """        
        # aggregate all the images into a list
        images = []
        plot_dir=os.path.join(self.output_dir, 'training_plots/')
        for i in range(1, len([e for e in os.listdir(plot_dir) if 'epoch_' in e])+1):
            filename = os.path.join(plot_dir, f'epoch_{i}.png')
            images.append(imageio.v3.imread(filename))

        # add a delay at the end
        for _ in range(int(len([e for e in os.listdir(plot_dir) if 'epoch_' in e])*0.05)):
            images.append(imageio.v3.imread(filename))
        
        # save the gif
        imageio.v3.imwrite(os.path.join(self.output_dir, 'training_process.gif'), images, loop=1000)
        
    def _save_model(self) -> IO:
        """
        Saves the trained pytorch model
        """
        config = {}
        config['arguments'] = self.model.arguments
        config['model'] = str(self.model.__class__)[8:-2].split('.')[-1]
        config['state_dict'] = self.model.to('cpu').state_dict()
        torch.save(config, os.path.join(self.output_dir, 'trained_model.' + config['model'] + '.pt'))
    
    def _save_encoder(self) -> IO:
        """
        Saves the trained backbone/encoder model
        """
        config = {}
        config['arguments'] = self.model.encoder.arguments
        config['model'] = str(self.model.encoder.__class__)[8:-2].split('.')[-1]
        config['state_dict'] = self.model.encoder.to('cpu').state_dict()
        torch.save(config, os.path.join(self.output_dir, 'trained_backbone.' + config['model'] + '.pt'))
    
    def fit(self, silent: bool = False) -> None:
        """
        Fit the model

        Args:
            silent (bool, optional): if True, suppress printing training progress
        
        Returns:
            None: stores models in .model attributes and, if specified, saves trained model to self.output_dir
        
        """        
        losses = [[], []]
        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            
            # training loop
            self.model.train()        
            for batch_idx, batch in enumerate(tqdm(self.train, desc='[Training loop]', disable=silent)):
                loss, xhat = self.training_step(batch[0].to(self.device))
                
            # validation loop
            self.model.eval()
            losses[0], losses[1] = losses[1], []
            for batch_idx, batch in enumerate(tqdm(self.valid, desc='[Validation loop]', disable=silent)):
                loss, xhat = self.validation_step(batch[0].to(self.device))
                losses[1].append(loss)
                self.logger.tally(batch_idx, [loss])
            
            # update logger
            self.logger.update(epoch)
            
            # update scheduler
            self.scheduler.step(self.logger.log['loss'][-1])
            
            # print metrics
            if not silent:
                self.logger.report()
            
            # generate plot of training progress
            self.generate_plots(batch[0], xhat, epoch, losses)
        
        # save output data
        if not isinstance(self.output_dir, type(None)):
            # generate a GIF of training process
            self.generate_gif()
            shutil.rmtree(os.path.join(self.output_dir, 'training_plots/'))        
            
            # save run information
            self.logger.save(os.path.join(self.output_dir, 'training_log.tsv'))
            
            # save the model state dict            
            self._save_model()
            
            # save the autoencoder backbone
            if self.save_backbone:
                self._save_encoder()