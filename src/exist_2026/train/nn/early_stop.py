from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, save_dir: str | Path, patience: int = 3, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.save_dir = Path(save_dir)
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module, val_probs_dict=None):
        if self.best_loss is None or val_loss < self.best_loss:
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss} --> {val_loss}). Saving...")
            self.best_loss = val_loss
            self.save_checkpoint(model, val_probs_dict)
            self.counter = 0
            return

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module, val_probs_dict):
        torch.save(model.state_dict(), self.save_dir / "best_model.pt")
        if val_probs_dict is not None:
            np.save(self.save_dir / "dl_val_probs.npy", val_probs_dict)
