import argparse
import sys
import time
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from utils import WikitextDataset, save_model_checkpoint, get_device, get_reduced_dataset, get_tokenizer, set_randomization_seed, get_optimizer, get_scheduler
from utils import TRAIN_DATA_FILE_PATH, VALID_DATA_FILE_PATH, TEST_DATA_FILE_PATH, DIFFUSION_FILE_PATH, SUBSET_RATIO_OF_DATASET_TO_TRAIN, SUBSET_RATIO_OF_DATASET_TO_TUNE
from utils import EARLY_STOP_EPOCHS, DIFFUSION_MODE_INDICATOR, BLOCK_SIZE_DIFFUSION, BATCH_SIZE_DIFFUSION, ACCUMULATION_STEPS_DIFFUSION

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--tune", type=int, default=0, help="Hyperparameter tuning mode level (0 means no tuning)")
    
    return parser.parse_args()

# A noise_ratio of 0.7 mean we swap 70% of the tokens
def forward_noise_process(tokens, vocab_size, noise_ratio, device):
    random_values = torch.rand(tokens.shape, device=device)
    mask = random_values < noise_ratio
    random_tokens = torch.randint(0, vocab_size, tokens.shape, device=device)
    forward_noise_tokens = torch.where(mask, random_tokens, tokens)
    return forward_noise_tokens

class DiffusionModel(nn.Module):
    def __init__(self, tokenizer, block_size, nhead, nhead_scalar, num_layers, time_embedding_dim, dropout, dim_feedforward_scalar, label_smoothing, max_timesteps=1000, self_conditioning_prob=0.5):
        super().__init__()

        self.max_timesteps = max_timesteps
        self.self_conditioning_prob = self_conditioning_prob

        # It must be possible to divide d_model by nhead, so I figured it would be best to create the d_model within the __init__.
        # Should make hyperparameter search a bit easier...
        d_model = nhead * nhead_scalar

        # Pull needed data from the tokenizer
        padding_index = tokenizer.pad_token_id
        vocab_size = len(tokenizer)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_index)
        self.positional_embedding = nn.Embedding(block_size, d_model)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward = dim_feedforward_scalar * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_head = nn.Linear(d_model, vocab_size)

        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_index, label_smoothing=label_smoothing)

    def forward(self, source_indices):
        batch_size, sequence_length = source_indices.shape

        

        return logits

def train_epoch(model, vocab_size, dataloader, optimizer, scheduler, accumulation_steps, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)

        x = forward_noise_process(x, vocab_size, 0.5, device)

        break

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
def train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, num_epochs=10,
                    n_layers=6, d_key_value=64, nhead=6, dim_feedforward_scalar=4, lr=5e-4,    # L1 Hyperparameters
                    warmup_pct_start=0.1, dropout=0.1, weight_decay=0.01, label_smoothing=0.1, # L2 Hyperparameters
                    trial=None):  # The trial for tuning.
    # Setup before initializing the model
    start_time = time.time()
    set_randomization_seed()
    vocab_size = len(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_DIFFUSION,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_DIFFUSION
    )

    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_DIFFUSION
        )

    # Model initialization
    model = DiffusionModel(tokenizer, BLOCK_SIZE_DIFFUSION, 
                                 d_key_value, nhead, n_layers, dropout, dim_feedforward_scalar,
                                 label_smoothing)
    model.to(device)

    optimizer = get_optimizer(model.parameters(), lr, weight_decay)
    scheduler = get_scheduler(optimizer, lr, num_epochs, train_loader, ACCUMULATION_STEPS_DIFFUSION, warmup_pct_start)

    # Model training
    training_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    no_improvement_epochs = 0
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_training_loss = train_epoch(model, vocab_size, train_loader, optimizer, scheduler, ACCUMULATION_STEPS_DIFFUSION, device)
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

def hyperparameter_tuning_objective_l1(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device):
    pass

def hyperparameter_tuning_objective_l2(trial, tokenizer, num_epochs, train_dataset, valid_dataset, device):
    pass

if __name__ == "__main__":
    device = get_device()
    tokenizer = get_tokenizer()

    # Prepare the datasets
    train_dataset = WikitextDataset(TRAIN_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR)
    valid_dataset = WikitextDataset(VALID_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR)

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
                                    study_name=f"Wikitext_Level_{DIFFUSION_MODE_INDICATOR}_{level}",
                                    storage=f"sqlite:///tuning_history_{DIFFUSION_MODE_INDICATOR}.db",
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
        test_dataset = WikitextDataset(TEST_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR)
        train_full_model(tokenizer, train_dataset, valid_dataset, test_dataset, device, 10
                         )