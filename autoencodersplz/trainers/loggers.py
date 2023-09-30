"""Classes for logging training and/or validation results"""

import logging
import pandas as pd
import tqdm as tqdm_module

class TqdmLoggingHandler(logging.Handler):
    """Enables writing to output without destroying tqdm progress bars"""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm_module.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
                        
class AutoencoderLogger:
    """Logs the validation results during training"""

    def __init__(self):
        self.log = {
            'epoch': [],
            'loss': []
        }        
        tqdm_module.tqdm._instances.clear()
        
        self.writer = logging.getLogger(__name__)
        self.writer.setLevel(logging.INFO)
        self.writer.addHandler(TqdmLoggingHandler())
        
    def tally(self, batch_idx: int, losses: tuple):
        """Tally the losses for the current epoch's validation step"""
        if batch_idx == 0:
            self.n = 0
            self.losses = [0 for _ in losses]
        
        self.n += 1
        self.losses = [self.losses[i] + losses[i] for i in range(len(losses))]
    
    def update(self, epoch: int):
        """Update the log with the current epoch's validation results"""
        self.log['epoch'].append(epoch)
        self.log['loss'].append(self.losses[0]/self.n)
        
    def report(self):        
        self.writer.info(
            f"Epoch {self.log['epoch'][-1]} results | " + \
            f"loss {self.log['loss'][-1]:.2e}"
        )
    
    def save(self, path: str):
        pd.DataFrame(self.log).to_csv(path, index=False, sep='\t')