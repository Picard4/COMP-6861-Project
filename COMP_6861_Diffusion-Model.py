import argparse
import sys
import random
import time
import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import Dataset, DataLoader
from utils import save_model_checkpoint, get_reduced_dataset, get_tokenizer, TRAIN_DATA_FILE_PATH, VALID_DATA_FILE_PATH, TEST_DATA_FILE_PATH, DIFFUSION_FILE_PATH, EARLY_STOP_EPOCHS, BLOCK_SIZE, SUBSET_RATIO_OF_DATASET_TO_TRAIN, SUBSET_RATIO_OF_DATASET_TO_TUNE

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--tune", type=int, default=0, help="Hyperparameter tuning mode level (0 means no tuning)")
    
    return parser.parse_args()

class WikitextDataset(Dataset):
    def __init__(self, file_path, block_size):
        # We need to start by loading data into CPU. We only move to GPU once batches are ready.
        self.data = torch.load(file_path, mmap=True, weights_only=True).long()
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        chunk = self.data[index : index + self.block_size]
        
        # The goal in a diffusion model is to remove noise to form the chunk, so the chunk is our target!
        return chunk

class BaselineDecoderModel(nn.Module):
    def __init__(self, tokenizer, block_size, d_key_value, nhead, n_layers, dropout, dim_feedforward_scalar, label_smoothing):
        super().__init__()

        # It must be possible to divide d_model by nhead, so I figured it would be best to create the d_model within the __init__.
        # Should make hyperparameter search a bit easier...
        d_model = nhead * d_key_value

        vocab_size = len(tokenizer)
        self.padding_index = tokenizer.pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_index)
        self.positional_embedding = nn.Embedding(block_size, d_model, padding_idx=self.padding_index)

        self.dropout = nn.Dropout(p=dropout)

        self.transformer_blocks = nn.ModuleList([
            # Technically this is meant to be a decoder, but the TransformerDecoderLayer is meant to use a mask for memory and target.
            # We only need a mask for the target (there is no memory since we have no encoder), so we can optimize out some math by using encoders.
            # Technically, this is still a decoder since it's autoregressive https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder
            nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward_scalar * d_model,
            batch_first=True,
            norm_first=True
            ) for layer in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_prediction_layer = nn.Linear(d_model, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.padding_index, label_smoothing=label_smoothing)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, source_indices):
        batch_size, sequence_length = source_indices.shape

        # The causal_mask from the __init__ is designed to handle a mask at maximum capacity... which is rarely what we need in practice!
        current_mask = self.causal_mask[:sequence_length, :sequence_length]

        token_embeddings = self.token_embedding(source_indices)
        positional_embeddings = self.positional_embedding(torch.arange(sequence_length, device=source_indices.device))

        x = self.dropout(token_embeddings + positional_embeddings)

        for block in self.transformer_blocks:
            x = block(
                src=x,
                src_mask=current_mask,
                is_causal=True
            )
        
        x = self.layer_norm(x)
        logits = self.linear_prediction_layer(x)

        return logits

def train_epoch(model, dataloader, optimizer, scheduler, accumulation_steps, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)

        # Use mixed-precision training to save some time!
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)

            loss = model.criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            # Scale the loss based on the gradient accumulation
            # The scaled loss affects our model, but we will use the unscaled loss to calculate the loss we want to return
            scaled_loss = loss / accumulation_steps

        scaled_loss.backward()
        
        # Use gradient accumulation to train more efficiently with less memory
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item()

        # Sleep every 10 batches to (hopefully) avoid microwaving my laptop
        if (i + 1) % 10 == 0:
            time.sleep(0.1)
            # To ensure that training is still working...
            if (i + 1) % 500 == 0:
                print(f"Batch {i+1}/{len(dataloader)}...", file=sys.stderr, flush=True)

    return total_loss / len(dataloader)

@torch.no_grad()
def eval_model(
    model,
    dataloader,
    device
):
    model.eval()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)

        # Use mixed-precision to save some time!
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)

            loss = model.criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

# This is the main code. Sets up the diffusion model.
# Datasets are assumed to use the same block_size as the datasets that are sent in.
# Send in None for the test_dataset to indicate that this training is for hyperparameter tuning.
# Block size must be adjusted on its own (since splitting the dataset is determined by block_size), so this project will omit tuning it.
# batch_size and accumulation_steps must go as high as my hardware will allow them to go (no tuning will be done for them as a result).
def train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, block_size=256, num_epochs=10,
                    n_layers=6, d_key_value=64, nhead=6, dim_feedforward_scalar=4, lr=5e-4,    # L1 Hyperparameters
                    warmup_pct_start=0.1, dropout=0.1, weight_decay=0.01, label_smoothing=0.1, # L2 Hyperparameters
                    batch_size=16, accumulation_steps=8, trial=None):  # Pre-determined hyperparameters (no tuning done) and the trial for tuning.
    # Setup before initializing the model
    start_time = time.time()
    random.seed(0)
    torch.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size
    )

    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )

    # Model initialization
    model = BaselineDecoderModel(tokenizer, block_size, 
                                 d_key_value, nhead, n_layers, dropout, dim_feedforward_scalar,
                                 label_smoothing)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Due to gradient accumulation, we don't call optimizer.step on every loop.
    # This is how many times we will *actually* call optimizer.step.
    total_effective_steps = num_epochs * ((len(train_loader) + accumulation_steps - 1) // accumulation_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_effective_steps,
        pct_start=warmup_pct_start
    )

    # Model training
    training_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    no_improvement_epochs = 0
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_training_loss = train_epoch(model, train_loader, optimizer, scheduler, accumulation_steps, device)
            epoch_valid_loss = eval_model(model, valid_loader, device)
            print(f"Epoch {epoch} | Training loss: {epoch_training_loss} | Valid loss: {epoch_valid_loss}", flush=True)

            training_losses.append(epoch_training_loss)
            valid_losses.append(epoch_valid_loss)
            if (epoch_valid_loss < best_valid_loss):
                best_valid_loss = epoch_valid_loss
                no_improvement_epochs = 0
                if test_dataset:
                    save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, DIFFUSION_FILE_PATH, f"BestModel.pt")
            else:
                no_improvement_epochs = no_improvement_epochs + 1
                if no_improvement_epochs >= EARLY_STOP_EPOCHS:
                    print(f"No improvements spotted for {EARLY_STOP_EPOCHS} epochs. Stopping early...", flush=True)
                    break
                if test_dataset:
                    save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, DIFFUSION_FILE_PATH, f"Checkpoint.pt")
            if trial is not None:
                trial.report(epoch_valid_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        if test_dataset:
            test_loss = eval_model(model, test_loader, device)
            print(f"Final Test Loss: {test_loss}", flush=True)
            end_time = time.time()
            total_duration = end_time - start_time
            print(f"Total Training Time: {total_duration / 3600:.2f} hours.", flush=True)
            torch.save({"training_losses": training_losses, "valid_losses": valid_losses, "test_loss": test_loss, "total_duration": total_duration}, DIFFUSION_FILE_PATH + f"Losses.pt")
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception thrown: {e}", file=sys.stderr, flush=True)
        if training_losses:
            save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, DIFFUSION_FILE_PATH, f"Crash-Checkpoint.pt")
            torch.save({"training_losses": training_losses, "valid_losses": valid_losses}, DIFFUSION_FILE_PATH + f"Losses.pt")
    finally:
        torch.cuda.empty_cache()
    # To help with hyperparameter tuning
    return best_valid_loss

def hyperparameter_tuning_objective_l1(trial, tokenizer, block_size, num_epochs, train_dataset, valid_dataset, device):
    pass

def hyperparameter_tuning_objective_l2(trial, tokenizer, block_size, num_epochs, train_dataset, valid_dataset, device):
    pass

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    tokenizer = get_tokenizer()

    # Prepare the datasets
    train_dataset = WikitextDataset(TRAIN_DATA_FILE_PATH, BLOCK_SIZE)
    valid_dataset = WikitextDataset(VALID_DATA_FILE_PATH, BLOCK_SIZE)

    # Reduce the training set to a manageable level
    train_dataset = get_reduced_dataset(train_dataset, SUBSET_RATIO_OF_DATASET_TO_TRAIN)

    args = get_args()
    if args.tune:
        level = min(args.tune, 2)
        print(f"Testing hyperparameter combinations at level {level}...", flush=True)
        train_dataset = get_reduced_dataset(train_dataset, SUBSET_RATIO_OF_DATASET_TO_TUNE)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(seed=42),
                                    study_name=f"Wikitext_Level_{level}",
                                    storage="sqlite:///tuning_history.db",
                                    load_if_exists=True,
                                    pruner=pruner)
        if args.tune == 1:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l1(trial, tokenizer, BLOCK_SIZE, 3, train_dataset, valid_dataset, device), 
                n_trials=12
            )
        else:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l2(trial, tokenizer, BLOCK_SIZE, 3, train_dataset, valid_dataset, device), 
                n_trials=8
            )
        print(f"Best Hyperparameters: {study.best_params}", flush=True)
    else:
        print("Training model...", flush=True)
        test_dataset = WikitextDataset(TEST_DATA_FILE_PATH, BLOCK_SIZE)
        train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, BLOCK_SIZE, 10
                         )