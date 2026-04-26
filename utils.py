import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizerFast

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
HYBRID_MODE_INDICATOR = "hybrid"

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
EARLY_STOP_EPOCHS = 2

# The main training dataset is about 1 GB.
# That's way too large to train in a reasonable time frame for me (one epoch on 1% takes roughly an hour to an hour + 45 minutes on my laptop). Let's cut it down to 2.5%.
# The tokenizer was trained on the full training set - this significantly reduces the risk of <unk> tokens, though, and it's the same between all models I will train.
# The validation and test sets are less than 1% of the training set - no need to reduce either.
SUBSET_RATIO_OF_DATASET_TO_TRAIN = 0.025

# Train on 20% of the reduced dataset - that means 0.5% of the original training set.
SUBSET_RATIO_OF_DATASET_TO_TUNE = 0.2

# Functions and classes  ----------------------------------------------------------


def set_randomization_seed():
    """
    Automatically sets the randomization seeds. To be called right before training a model.
    """
    random.seed(0)
    torch.manual_seed(0)


def get_device():
    """
    Attempts to get and return a cuda device. If that fails, returns the cpu.

    Returns:
        The chosen device.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    return device


def get_tokenizer():
    """
    Loads the tokenizer, adds the tokens it needs to process special characters, and returns it.

    Returns:
        The tokenizer.
    """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE_PATH)

    # We need to ensure the tokenizer knows what the special tokens mean.
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "eos_token": "<eos>",
            "mask_token": "<mask>",
        }
    )

    return tokenizer


def get_reduced_dataset(full_dataset, subset_ratio=0.01):
    """
    Reduces the sent dataset by the specified subset_ratio.
    A subset ratio of 0.01 will reduce the dataset to 1% of what it originally was.
    Reduction is done deterministically to ensure consistency when training models.

    Args:
        full_dataset (WikitextDataset): The original dataset to be reduced.
        subset_ratio (float): The ratio of which to reduce the dataset.

    Returns:
        The reduced dataset.
    """
    subset_size = int(len(full_dataset) * subset_ratio)
    indices = np.arange(len(full_dataset))

    # rng is hard-coded to ensure deterministic behaviour
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    subset_indices = indices[:subset_size]
    return Subset(full_dataset, subset_indices)


def get_optimizer(parameters, lr, weight_decay):
    """
    Gets a newly instantiated AdamW optimizer.

    Args:
        lr (float): The learning rate.
        weight_decay (float): The weight decay.

    Returns:
        The newly instantiated AdamW optimizer.
    """
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer, lr, num_epochs, train_loader, accumulation_steps, warmup_pct_start
):
    """
    Gets a newly instantiated OneCycleLR scheduler.

    Args:
        optimizer (torch.optim.AdamW): The model's optimizer.
        lr (float): The model's learning rate.
        num_epochs (int): The number of epochs that the model will be trained.
        train_loader (DataLoader): The DataLoader for the training set.
        accumulation_steps (int): The number of steps to take before calling optimizer.step.
        warmup_pct_start (float): The percentage that the model will spend with a "warmup" for its learning rate.

    Returns:
        The newly instantiated OneCycleLR scheduler.
    """
    # Due to gradient accumulation, we don't call optimizer.step on every loop.
    # This is how many times we will *actually* call optimizer.step.
    total_effective_steps = num_epochs * (
        (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    )

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_effective_steps,
        pct_start=warmup_pct_start,
    )


def save_model_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    epoch_training_loss,
    best_valid_loss,
    file_path,
    save_file_name,
):
    """
    Saves a checkpoint of the model and its recent performance.

    Args:
        epoch (int): The epoch that just concluded.
        model (nn.Module): The model that is training.
        optimizer (torch.optim.AdamW): The model's optimizer.
        scheduler (torch.optim.lr_scheduler.OneCycleLR): The model's scheduler.
        epoch_training_loss (float): The latest epoch's training loss.
        best_valid_loss (float): The model's best validation loss so far.
        file_path (string): The path to where the model should be saved.
        save_file_name (string): The name of the file to save the model to.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_training_loss": epoch_training_loss,
        "best_valid_loss": best_valid_loss,
    }

    torch.save(checkpoint, file_path + save_file_name)


class WikitextDataset(Dataset):
    """
    A dataset that manages training, validation, or test data.
    """

    def __init__(self, file_path, block_size, mode=BASELINE_MODE_INDICATOR):
        """
        Initializes the Dataset.

        Args:
            file_path (string): The full file path to load the data from.
            block_size (int): The block size that the Dataset will use to get items.
            mode (string): The mode that the Dataset will use to get items - it must match the model that it will be used on.
        """
        # We need to start by loading data into CPU. We only move to GPU once batches are ready.
        self.data = torch.load(file_path, mmap=True, weights_only=True).long()
        self.block_size = block_size
        self.mode = mode

    def __len__(self):
        """
        Gets the length of the Dataset, which is the length of its data minus a buffer representing its block size.

        Returns:
            The length of the Dataset.
        """
        return len(self.data) - DATASET_BUFFER

    def __getitem__(self, index):
        """
        Gets the next item in the dataset, based on the specified index.

        Args:
            index (int): The index to search the Dataset's data to get the next item.

        Returns:
            A tuple containing the next item. Index 0 contains the input, and index 1 contains the target.
        """
        chunk = self.data[index : index + self.block_size]

        if self.mode == BASELINE_MODE_INDICATOR:
            # The goal is to predict the next token, so we just need to shift our chunk by one.
            target = self.data[index + 1 : index + self.block_size + 1]
            return chunk, target
        if self.mode == DIFFUSION_MODE_INDICATOR:
            # The goal in a diffusion model is to remove noise to form the chunk, so the chunk is our target!
            return chunk, chunk
        if self.mode == HYBRID_MODE_INDICATOR:
            # The hybrid model will generate half the tokens indicated by the block size, then denoise them.
            # As such, we want to focus on evaluating the generated tokens only.
            half_block_size = self.block_size // 2
            target = self.data[
                index + self.block_size : index + self.block_size + half_block_size
            ]
            return chunk, target
