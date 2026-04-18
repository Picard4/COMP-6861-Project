import torch
import numpy as np
from torch.utils.data import Subset
from transformers import PreTrainedTokenizerFast

# File constants
DATA_FILE_PATH = "save-data/"
TOKENIZER_FILE_PATH = DATA_FILE_PATH + "wikitext-tokenizer.json"
TRAIN_DATA_FILE_PATH = DATA_FILE_PATH + "train-dataset-tokens.pt"
VALID_DATA_FILE_PATH = DATA_FILE_PATH + "valid-dataset-tokens.pt"
TEST_DATA_FILE_PATH = DATA_FILE_PATH + "test-dataset-tokens.pt"
BASELINE_FILE_PATH = DATA_FILE_PATH + "baseline-models/"
DIFFUSION_FILE_PATH = DATA_FILE_PATH + "diffusion-models/"

# Block size must be adjusted on its own, if at all, since splitting the dataset is determined by block_size.
# Block size must stay the same between all models to ensure the split dataset is always the same dataset.
# 512 may be an alternative to consider, given time...
BLOCK_SIZE=256

# Number of epochs to allow no validation loss improvement before performing early stopping.
EARLY_STOP_EPOCHS = 3

# The main training dataset is about 1 GB. 
# That's way too large to train in a reasonable time frame for me (one epoch on 1% takes roughly an hour to an hour + 45 minutes). Let's cut it down to 2.5%.
# The tokenizer was trained on the full training set - this significantly reduces the risk of <unk> tokens, though, and it's the same between all models I will train.
# The validation and test sets are less than 1% of the training set - no need to reduce either.
SUBSET_RATIO_OF_DATASET_TO_TRAIN = 0.025

# Train on 20% of the reduced dataset - that means 0.5% of the original training set.
SUBSET_RATIO_OF_DATASET_TO_TUNE = 0.2

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

def get_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE_PATH)

    # We need to ensure the tokenizer knows what the special tokens mean.
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "eos_token": "<eos>",
        "mask_token": "<mask>"
    })

    return tokenizer