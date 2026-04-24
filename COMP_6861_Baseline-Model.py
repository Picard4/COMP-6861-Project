import argparse
import sys
import time
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from utils import WikitextDataset, save_model_checkpoint, get_device, get_reduced_dataset, get_tokenizer, set_randomization_seed, get_optimizer, get_scheduler
from utils import TRAIN_DATA_FILE_PATH, VALID_DATA_FILE_PATH, TEST_DATA_FILE_PATH, BASELINE_FILE_PATH, SUBSET_RATIO_OF_DATASET_TO_TRAIN, SUBSET_RATIO_OF_DATASET_TO_TUNE
from utils import EARLY_STOP_EPOCHS, BASELINE_MODE_INDICATOR, BLOCK_SIZE_BASELINE, BATCH_SIZE_BASELINE, ACCUMULATION_STEPS_BASELINE

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--tune", type=int, default=0, help="Hyperparameter tuning mode level (0 means no tuning)")
    
    return parser.parse_args()

class BaselineDecoderModel(nn.Module):
    def __init__(self, tokenizer, block_size, d_key_value, nhead, n_layers, dropout, dim_feedforward_scalar, label_smoothing):
        super().__init__()

        # It must be possible to divide d_model by nhead, so I figured it would be best to create the d_model within the __init__.
        # Should make hyperparameter search a bit easier...
        d_model = nhead * d_key_value

        vocab_size = len(tokenizer)
        self.padding_index = tokenizer.pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_index)
        self.positional_embedding = nn.Embedding(block_size, d_model, padding_idx=self.padding_index) # The padding_idx is a bug. I lack the time to retrain my model.

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

# This is the main code. Sets up the baseline model.
# Datasets are assumed to use the same block_size as the datasets that are sent in.
# Send in None for the test_dataset to indicate that this training is for hyperparameter tuning.
def train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, num_epochs=10,
                    n_layers=6, d_key_value=64, nhead=6, dim_feedforward_scalar=4, lr=5e-4,    # L1 Hyperparameters
                    warmup_pct_start=0.1, dropout=0.1, weight_decay=0.01, label_smoothing=0.1, # L2 Hyperparameters
                    trial=None):  # The trial for tuning.
    # Setup before initializing the model
    start_time = time.time()
    set_randomization_seed()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_BASELINE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_BASELINE
    )

    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_BASELINE
        )

    # Model initialization
    model = BaselineDecoderModel(tokenizer, BLOCK_SIZE_BASELINE, 
                                 d_key_value, nhead, n_layers, dropout, dim_feedforward_scalar,
                                 label_smoothing)
    model.to(device)

    optimizer = get_optimizer(model.parameters(), lr, weight_decay)
    scheduler = get_scheduler(optimizer, lr, num_epochs, train_loader, ACCUMULATION_STEPS_BASELINE, warmup_pct_start)

    # Model training
    training_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    no_improvement_epochs = 0
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_training_loss = train_epoch(model, train_loader, optimizer, scheduler, ACCUMULATION_STEPS_BASELINE, device)
            epoch_valid_loss = eval_model(model, valid_loader, device)
            print(f"Epoch {epoch} | Training loss: {epoch_training_loss} | Valid loss: {epoch_valid_loss}", flush=True)

            training_losses.append(epoch_training_loss)
            valid_losses.append(epoch_valid_loss)
            if (epoch_valid_loss < best_valid_loss):
                best_valid_loss = epoch_valid_loss
                no_improvement_epochs = 0
                if test_dataset:
                    save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, BASELINE_FILE_PATH, f"BestModel.pt")
            else:
                no_improvement_epochs = no_improvement_epochs + 1
                if no_improvement_epochs >= EARLY_STOP_EPOCHS:
                    print(f"No improvements spotted for {EARLY_STOP_EPOCHS} epochs. Stopping early...", flush=True)
                    break
                if test_dataset:
                    save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, BASELINE_FILE_PATH, f"Checkpoint.pt")
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
            torch.save({"training_losses": training_losses, "valid_losses": valid_losses, "test_loss": test_loss, "total_duration": total_duration}, BASELINE_FILE_PATH + f"Losses.pt")
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception thrown: {e}", file=sys.stderr, flush=True)
        if training_losses:
            save_model_checkpoint(epoch, model, optimizer, scheduler, epoch_training_loss, best_valid_loss, BASELINE_FILE_PATH, f"Crash-Checkpoint.pt")
            torch.save({"training_losses": training_losses, "valid_losses": valid_losses}, BASELINE_FILE_PATH + f"Losses.pt")
    finally:
        torch.cuda.empty_cache()
    # To help with hyperparameter tuning
    return best_valid_loss

def hyperparameter_tuning_objective_l1(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device):
    n_layers = trial.suggest_int("n_layers", 4, 12)
    d_key_value = trial.suggest_categorical("d_key_value", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [4, 8, 12])
    dim_feedforward_scalar = trial.suggest_int("dim_feedforward_scalar", 2, 4)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # These two hyperparameters make up the d_model.
    # If it's too high, my training will crash!
    if (nhead * d_key_value) > 768:
        raise optuna.exceptions.TrialPruned()

    validation_loss = train_full_model(tokenizer, train_dataset, valid_dataset, None, device, num_epochs,
                                       n_layers=n_layers, d_key_value=d_key_value, nhead=nhead, dim_feedforward_scalar=dim_feedforward_scalar, lr=lr,
                                       trial=trial)
    return validation_loss

def hyperparameter_tuning_objective_l2(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device):
    warmup_pct_start = trial.suggest_float("warmup_pct_start", 0.05, 0.3)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    validation_loss = train_full_model(tokenizer, train_dataset, valid_dataset, None, device, num_epochs,
                                       n_layers=9, d_key_value=32, nhead=12, dim_feedforward_scalar=2, lr=0.003063462210622081, 
                                       warmup_pct_start=warmup_pct_start, dropout=dropout, weight_decay=weight_decay, label_smoothing=label_smoothing,
                                       trial=trial)
    return validation_loss

if __name__ == "__main__":
    device = get_device()
    tokenizer = get_tokenizer()

    # Prepare the datasets
    train_dataset = WikitextDataset(TRAIN_DATA_FILE_PATH, BLOCK_SIZE_BASELINE, mode=BASELINE_MODE_INDICATOR)
    valid_dataset = WikitextDataset(VALID_DATA_FILE_PATH, BLOCK_SIZE_BASELINE, mode=BASELINE_MODE_INDICATOR)

    # Reduce the training set to a manageable level
    train_dataset = get_reduced_dataset(train_dataset, SUBSET_RATIO_OF_DATASET_TO_TRAIN)

    args = get_args()
    if args.tune:
        level = min(args.tune, 2)
        num_epochs = 3
        print(f"Testing hyperparameter combinations at level {level}...", flush=True)
        train_dataset = get_reduced_dataset(train_dataset, SUBSET_RATIO_OF_DATASET_TO_TUNE)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(seed=42),
                                    study_name=f"Wikitext_Level_{BASELINE_MODE_INDICATOR}_{level}",
                                    storage=f"sqlite:///tuning_history_{BASELINE_MODE_INDICATOR}.db",
                                    load_if_exists=True,
                                    pruner=pruner)
        if args.tune == 1:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l1(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device), 
                n_trials=12
            )
        else:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l2(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device), 
                n_trials=8
            )
        print(f"Best Hyperparameters: {study.best_params}", flush=True)
    else:
        print("Training model...", flush=True)
        test_dataset = WikitextDataset(TEST_DATA_FILE_PATH, BLOCK_SIZE_BASELINE, mode=BASELINE_MODE_INDICATOR)
        train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, 10,
                         n_layers=9, d_key_value=32, nhead=12, dim_feedforward_scalar=2, lr=1e-3,
                         warmup_pct_start=0.1, dropout=0.0662576444519992, weight_decay=0.012957079329680455, label_smoothing=0.03410482473745831)