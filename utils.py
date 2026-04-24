import torch
import random
import numpy as np
from torch.utils.data import Subset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset

# File constants ----------------------------------------------------------
DATA_FILE_PATH = "save-data/"
TOKENIZER_FILE_PATH = DATA_FILE_PATH + "wikitext-tokenizer.json"
TRAIN_DATA_FILE_PATH = DATA_FILE_PATH + "train-dataset-tokens.pt"
VALID_DATA_FILE_PATH = DATA_FILE_PATH + "valid-dataset-tokens.pt"
TEST_DATA_FILE_PATH = DATA_FILE_PATH + "test-dataset-tokens.pt"
BASELINE_FILE_PATH = DATA_FILE_PATH + "baseline-models/"
DIFFUSION_FILE_PATH = DATA_FILE_PATH + "diffusion-models/"

# Dataset constants ----------------------------------------------------------
BASELINE_MODE_INDICATOR = "baseline"
DIFFUSION_MODE_INDICATOR = "diffusion"

# Constants that must remain the same between models ----------------------------------------------------------

# Block size must be adjusted on its own, if at all, since splitting the dataset is determined by block_size.
# Since Block sizes affect the datasets, they are stored here.
# Block sizes also determine the context each model can access, so I think it's best to keep them the same at a safe value for each.
# Just in case I need to change that, they're separate variables.
BLOCK_SIZE_BASELINE = 128
BLOCK_SIZE_DIFFUSION = BLOCK_SIZE_BASELINE

# The length of our datasets cannot be controlled by the block size (just in case we want to make it different between models).
# So we use this value to fill that role instead.
# The value must be equal to or higher than the largest block size.
DATASET_BUFFER = 128

# BATCH_SIZE * ACCUMULATION_STEPS must remain the same between models, but the values of each can change.
# batch_size and accumulation_steps must go as high as my hardware will allow them to go.
EFFECTIVE_BATCH_SIZE = 128

BATCH_SIZE_BASELINE = 16
ACCUMULATION_STEPS_BASELINE = EFFECTIVE_BATCH_SIZE // BATCH_SIZE_BASELINE

BATCH_SIZE_DIFFUSION = 16
ACCUMULATION_STEPS_DIFFUSION = EFFECTIVE_BATCH_SIZE // BATCH_SIZE_DIFFUSION

# Number of epochs to allow no validation loss improvement before performing early stopping.
EARLY_STOP_EPOCHS = 3

# The main training dataset is about 1 GB. 
# That's way too large to train in a reasonable time frame for me (one epoch on 1% takes roughly an hour to an hour + 45 minutes). Let's cut it down to 2%.
# The tokenizer was trained on the full training set - this significantly reduces the risk of <unk> tokens, though, and it's the same between all models I will train.
# The validation and test sets are less than 1% of the training set - no need to reduce either.
SUBSET_RATIO_OF_DATASET_TO_TRAIN = 0.02

# Train on 20% of the reduced dataset - that means 0.4% of the original training set.
SUBSET_RATIO_OF_DATASET_TO_TUNE = 0.2

# Functions and classes  ----------------------------------------------------------

def set_randomization_seed():
    random.seed(0)
    torch.manual_seed(0)

def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    return device

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

def get_reduced_dataset(full_dataset, subset_ratio=0.01):
    subset_size = int(len(full_dataset) * subset_ratio)
    indices = np.arange(len(full_dataset))

    # rng is hard-coded to ensure deterministic behaviour
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    subset_indices = indices[:subset_size]
    return Subset(full_dataset, subset_indices)

def get_optimizer(parameters, lr, weight_decay):
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, lr, num_epochs, train_loader, accumulation_steps, warmup_pct_start):
    # Due to gradient accumulation, we don't call optimizer.step on every loop.
    # This is how many times we will *actually* call optimizer.step.
    total_effective_steps = num_epochs * ((len(train_loader) + accumulation_steps - 1) // accumulation_steps)

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_effective_steps,
        pct_start=warmup_pct_start
    )

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

class WikitextDataset(Dataset):
    def __init__(self, file_path, block_size, mode=BASELINE_MODE_INDICATOR):
        # We need to start by loading data into CPU. We only move to GPU once batches are ready.
        self.data = torch.load(file_path, mmap=True, weights_only=True).long()
        self.block_size = block_size
        self.mode = mode
    
    def __len__(self):
        return len(self.data) - DATASET_BUFFER
    
    def __getitem__(self, index):
        chunk = self.data[index : index + self.block_size]

        if self.mode == BASELINE_MODE_INDICATOR:
            # The goal is to predict the next token, so we just need to shift our chunk by one.
            target = self.data[index + 1 : index + self.block_size + 1]
            return chunk, target
        if self.mode == DIFFUSION_MODE_INDICATOR:
            # The goal in a diffusion model is to remove noise to form the chunk, so the chunk is our target!
            return chunk