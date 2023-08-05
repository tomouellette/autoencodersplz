import math
import warnings
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class CosineDecayWarmUp(_LRScheduler):
    """
    Implementing cosine decay with warmup learning rate scheduling using torch base class  

    Args:
        optimizer (torch.optim.Optimizer): instantiated torch optimizer
        epochs (int): maximum number of training epochs
    
    """
    def __init__(
            self, 
            optimizer, 
            epochs: int, 
            min_lr: float = 0.0001, 
            warmup_epochs: int = 40, 
            last_epoch: int = -1, 
            verbose: bool = False
        ):
        super(CosineDecayWarmUp, self).__init__(optimizer, last_epoch, verbose)
        if min_lr <= 0 or not isinstance(min_lr, float):
            raise ValueError("Expected positive float min_lr, but got {}".format(min_lr))
        if warmup_epochs < 1 or not isinstance(warmup_epochs, int):
            raise ValueError("Expected integer warmup_epochs >= 1, but got {}".format(warmup_epochs))
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.epochs = epochs
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.epoch < self.warmup_epochs:
            return [base_lr * self.epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * \
                    (1. + math.cos(math.pi * (self.epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs))) \
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Step could be called after every batch update
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        
        self.epoch = self.last_epoch + 1
        self.last_epoch = math.floor(self.epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]