import torch
import numpy as np
from torch.utils.data import Subset

# File constants
DATA_FILE_PATH = "save-data/"
TOKENIZER_FILE_PATH = DATA_FILE_PATH + "wikitext-tokenizer.json"
TRAIN_DATA_FILE_PATH = DATA_FILE_PATH + "train-dataset-tokens.pt"
VALID_DATA_FILE_PATH = DATA_FILE_PATH + "valid-dataset-tokens.pt"
TEST_DATA_FILE_PATH = DATA_FILE_PATH + "test-dataset-tokens.pt"
BASELINE_FILE_PATH = DATA_FILE_PATH + "baseline-models/"
DIFFUSION_FILE_PATH = DATA_FILE_PATH + "diffusion-models/"

def save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, file_path, save_file_name):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_training_loss": epoch_training_loss,
        "best_valid_loss": best_valid_loss
    }

    torch.save(checkpoint, file_path + save_file_name)

def get_reduced_dataset(full_dataset, subset_ratio=0.01):
    subset_size = int(len(full_dataset) * subset_ratio)
    indices = np.arange(len(full_dataset))

    # rng is hard-coded to ensure deterministic behaviour
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    subset_indices = indices[:subset_size]
    return Subset(full_dataset, subset_indices)