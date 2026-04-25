import argparse
import math
import sys
import time

import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from utils import (
    ACCUMULATION_STEPS_DIFFUSION,
    BATCH_SIZE_DIFFUSION,
    BLOCK_SIZE_DIFFUSION,
    DIFFUSION_FILE_PATH,
    DIFFUSION_MODE_INDICATOR,
    EARLY_STOP_EPOCHS,
    SUBSET_RATIO_OF_DATASET_TO_TRAIN,
    SUBSET_RATIO_OF_DATASET_TO_TUNE,
    TEST_DATA_FILE_PATH,
    TRAIN_DATA_FILE_PATH,
    VALID_DATA_FILE_PATH,
    WikitextDataset,
    get_device,
    get_optimizer,
    get_reduced_dataset,
    get_scheduler,
    get_tokenizer,
    save_model_checkpoint,
    set_randomization_seed,
)

LINEAR_NOISE_SCHEDULE = "linear"
COSINE_NOISE_SCHEDULE = "cosine"


def get_args():
    """
    Gets the arguments that this model will use when run directly.
    The only argument is "tune", which defines the level of hyperparameter tuning to do. The values are as follows:
    - A value of 0 will train the model rather than do any tuning.
    - A value of 1 will do tuning on the model's architectural hyperparameters.
    - A value of 2 will do tuning on the model's lesser parameters (mainly regularization parameters).

    Returns:
        The namespace of arguments that this program will use.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tune",
        type=int,
        default=0,
        help="Hyperparameter tuning mode level (0 means no tuning)",
    )

    return parser.parse_args()


def get_noise_schedule(max_timesteps, schedule_type):
    """
    Gets the requested noise schedule for the diffusion model.

    Args:
        max_timesteps (int): The max_timesteps that the diffusion model will use.
        schedule_type (string): The type of noise schedule that is desired (as a string literal constant). "linear" and "cosine" are currently supported.

    Returns:
        The requested noise schedule.
    """
    if schedule_type == LINEAR_NOISE_SCHEDULE:
        return torch.linspace(0, 1, max_timesteps)
    elif schedule_type == COSINE_NOISE_SCHEDULE:
        steps = torch.arange(max_timesteps)
        return 1.0 - torch.cos((steps / max_timesteps) * (math.pi / 2))


def forward_noise_process(tokens, vocab_size, noise_ratio, padding_index, device):
    """
    Adds noise to the sent tensor of tokens by randomly changing some of the tokens to other tokens.

    Args:
        tokens (torch.Tensor): The tensor of tokens to add noise to.
        vocab_size (int): The size of the model's vocaculary.
        noise_ratio (float or torch.Tensor): The ratio of approximately how many tokens in the tensor should be distorted.
            A noise_ratio of 0.7 will apply a mask which will swap roughly 70% of the tokens.
            Supports a single float or a tensor of floats in case multiple noise_ratios are preferred for a single batch.
        padding_index (int): The index to the padding token.
        device (DeviceLikeType): The device that is being used to train the model.

    Returns:
        A noisy version of the sent tokens tensor, where some tokens have been swapped with random tokens.
    """
    random_values = torch.rand(tokens.shape, device=device)
    mask = (random_values < noise_ratio) & (tokens != padding_index)
    random_tokens = torch.randint(0, vocab_size, tokens.shape, device=device)
    forward_noise_tokens = torch.where(mask, random_tokens, tokens)
    return forward_noise_tokens


class DiffusionModel(nn.Module):
    """
    A model that learns how to edit an existing sequence of characters by changing tokens that were swapped to a random token back into the tokens they are meant to be.
    """

    def __init__(
        self,
        tokenizer,
        block_size,
        nhead,
        nhead_scalar,
        num_layers,
        time_embedding_dim,
        dropout,
        dim_feedforward_scalar,
        label_smoothing,
        max_timesteps,
    ):
        """
        Initializes the DiffusionModel with all of the components that it needs, including a criterion function (Categorical Cross-Entropy).

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer that the model is training to work with.
            block_size (int): The amount of context the model must use to denoise the sequence. If set to 128, the model has 128 tokens to evaluate.
            nhead (int): The model's number of multi-head attention heads.
            nhead_scalar (int): The d_model of the model is formed by multiplying this with nhead.
            num_layers (int): The number of Transformer encoder layers to use with the model.
            time_embedding_dim (int): The d_model for the time_embedding neural network in the model.
            dropout (float): The dropout ratio to use for training.
            dim_feedforward_scalar (int): The scalar to multiply with the model's d_model to form the dim_feedforward of each Transformer block in the model.
            label_smoothing (float): The ratio to smooth the model's labels, making it less confident in its answers.
            max_timesteps (int): The model's maximum number of time steps for adding and removing noise.
        """
        super().__init__()

        self.max_timesteps = max_timesteps

        # It must be possible to divide d_model by nhead, so I figured it would be best to create the d_model within the __init__.
        # Should make hyperparameter search a bit easier...
        d_model = nhead * nhead_scalar

        # Pull needed data from the tokenizer
        padding_index = tokenizer.pad_token_id
        vocab_size = len(tokenizer)

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_index
        )
        self.positional_embedding = nn.Embedding(block_size, d_model)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, d_model),
        )

        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward_scalar * d_model,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for layer in range(num_layers)
            ]
        )

        # These blocks ensure that the time information attached to the Transformer block input is up-to-date
        self.time_projection_blocks = nn.ModuleList(
            [nn.Linear(d_model, d_model) for layer in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_head = nn.Linear(d_model, vocab_size)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=padding_index, label_smoothing=label_smoothing
        )
        self._initialize_weights()

    def forward(self, source_indices, time_steps):
        """
        Predicts the correct sequence of tokens for the each sequence in the noisy source_indices.
        Uses the time_steps to determine how noisy the sequence tokens in source_indices are.

        Args:
            source_indices (torch.Tensor): A tensor of token sequences. Some of these tokens were swapped to random tokens in a forward noise process.
            time_steps (torch.Tensor): A tensor of time step values - the higher the value, the more noise was added to the source_indices.

        Returns:
            The model's predicted logits.
        """
        batch_size, sequence_length = source_indices.shape

        token_embeddings = self.token_embedding(source_indices)
        positional_embeddings = self.positional_embedding(
            torch.arange(sequence_length, device=source_indices.device)
        ).unsqueeze(0)
        time_information = self.time_embedding(
            time_steps.float().view(-1, 1) / self.max_timesteps
        ).unsqueeze(1)

        x = token_embeddings + positional_embeddings
        for i, block in enumerate(self.transformer_blocks):
            # Inject the time information for each block to keep the model aware of how many steps it has.
            x = block(x + self.time_projection_blocks[i](time_information))
        x = self.layer_norm(x)
        logits = self.linear_head(x)

        return logits

    def _initialize_weights(self):
        """
        Initalizes the weights of the diffusion model using zero-initialization.
        This approach helps the model learn language before it needs to learn diffusion.
        Taken from: https://openreview.net/forum?id=E4roJSM9RM
        """
        # Initialize the last layer of the time embedding to zero.
        nn.init.zeros_(self.time_embedding[-1].weight)
        nn.init.zeros_(self.time_embedding[-1].bias)

        # Initialize every time projection layer to zero.
        for projection in self.time_projection_blocks:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)


def train_epoch(
    model,
    vocab_size,
    dataloader,
    optimizer,
    scheduler,
    accumulation_steps,
    noise_schedule,
    max_timesteps,
    padding_index,
    device,
):
    """
    Trains the sent diffusion model for one epoch.

    Args:
        model (DiffusionModel): The model to train for one epoch.
        vocab_size (int): The size of the model's vocabulary.
        dataloader (DataLoader): The training dataset's dataloader.
        optimizer (torch.optim.AdamW): The model's optimizer.
        scheduler (torch.optim.lr_scheduler.OneCycleLR): The model's scheduler.
        accumulation_steps (int): The number of steps to take before calling optimizer.step.
        noise_schedule (torch.Tensor): A tensor containing a veriety of noise_ratios to use for the forward noise process.
        max_timesteps (int): The maximum number of time steps that can be used to add or remove noise.
        padding_index (int): The index to the padding token.
        device (DeviceLikeType): The device that is being used to train the model.

    Returns:
        The model's training loss for this epoch.
    """
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)

        # We have batch_size sequences of block_size tokens, so the dimensionality is (batch_size, block_size)
        # We want a different noise_ratio for each block - that way the model gets more exposure to different denoising problems.
        time_steps = torch.randint(0, max_timesteps, (x.size(0),), device=device)
        noise_ratios = noise_schedule[time_steps].view(-1, 1)
        x = forward_noise_process(x, vocab_size, noise_ratios, padding_index, device)

        # Use mixed-precision training to save some time!
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x, time_steps)

            loss = model.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

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

        # To ensure that training is still working...
        if (i + 1) % 1000 == 0:
            print(f"Batch {i + 1}/{len(dataloader)}...", file=sys.stderr, flush=True)

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_model(
    model, vocab_size, dataloader, noise_schedule, max_timesteps, padding_index, device
):
    """
    Performs a round of evaluation for the sent model on a validation or test set.

    Args:
        model (DiffusionModel): The model to evaluate.
        vocab_size (int): The size of the model's vocabulary.
        dataloader (DataLoader): The validation or test dataset's dataloader.
        noise_schedule (torch.Tensor): A tensor containing a veriety of noise_ratios to use for the forward noise process.
        max_timesteps (int): The maximum number of time steps that can be used to add or remove noise.
        padding_index (int): The index to the padding token.
        device (DeviceLikeType): The device that is being used to train the model.

    Returns:
        The model's validation or test loss for an epoch.
    """
    model.eval()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)

        time_steps = torch.randint(0, max_timesteps, (x.size(0),), device=device)
        noise_ratios = noise_schedule[time_steps].view(-1, 1)
        x = forward_noise_process(x, vocab_size, noise_ratios, padding_index, device)

        # Use mixed-precision to save some time!
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x, time_steps)

            loss = model.criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def train_full_model(
    tokenizer,
    train_dataset,
    valid_dataset,
    test_dataset,
    device,
    num_epochs=10,
    lr=5e-4,
    max_timesteps=1000,  # Hyperparameters that will not be in the search
    nhead=6,
    nhead_scalar=48,
    num_layers=6,
    time_embedding_dim=96,  # L1 Hyperparameters
    dim_feedforward_scalar=4,
    noise_schedule_type=COSINE_NOISE_SCHEDULE,  # L1 Hyperparameters
    warmup_pct_start=0.1,
    dropout=0.1,
    weight_decay=0.05,
    label_smoothing=0.1,  # L2 Hyperparameters (if time allows)
    trial=None,
):  # The trial for tuning.
    """
    Trains a full diffusion model, from start to end, saving the model's progress to external files along the way.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer that the model is training to work with.
        train_dataset (WikitextDataset): The model's training dataset.
        valid_dataset (WikitextDataset): The model's validation dataset.
        test_dataset (WikitextDataset): The model's test dataset.
            Send None to indicate that this training is for hyperparameter tuning.
        device (DeviceLikeType): The device that is being used to train the model.
        num_epochs (int): The number of epochs to train the model for.
        lr (float): The model's learning rate hyperparameter.
        max_timesteps (int): The model's maximum number of time steps for adding and removing noise.
        nhead (int): The model's number of multi-head attention heads.
        nhead_scalar (int): The d_model of the model is formed by multiplying this with nhead.
        num_layers (int): The number of Transformer encoder layers to use with the model.
        time_embedding_dim (int): The d_model for the time_embedding neural network in the model.
        dim_feedforward_scalar (int): The scalar to multiply with the model's d_model to form the dim_feedforward of each Transformer block in the model.
        noise_schedule_type (string): The type of noise schedule to use for training and evaluation.
        warmup_pct_start (float): The ratio of the model's training time to use a warmup scheduler.
        dropout (float): The dropout ratio to use for training.
        weight_decay (float): The ratio for decaying the model's weights before updating them to help prevent overfitting.
        label_smoothing (float): The ratio to smooth the model's labels, making it less confident in its answers.
        trial (Any): The trial object Optuna uses to keep track of the model as it trains.

    Returns:
        The model's best validation loss (to help with hyperparameter tuning).
    """
    # Setup before initializing the model
    start_time = time.time()
    set_randomization_seed()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_DIFFUSION,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_DIFFUSION)

    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_DIFFUSION)

    # Model initialization
    model = DiffusionModel(
        tokenizer,
        BLOCK_SIZE_DIFFUSION,
        nhead,
        nhead_scalar,
        num_layers,
        time_embedding_dim,
        dropout,
        dim_feedforward_scalar,
        label_smoothing,
        max_timesteps,
    )
    model.to(device)

    optimizer = get_optimizer(model.parameters(), lr, weight_decay)
    scheduler = get_scheduler(
        optimizer,
        lr,
        num_epochs,
        train_loader,
        ACCUMULATION_STEPS_DIFFUSION,
        warmup_pct_start,
    )
    noise_schedule = get_noise_schedule(max_timesteps, noise_schedule_type).to(device)

    # Model training
    training_losses = []
    valid_losses = []
    best_valid_loss = float("inf")
    no_improvement_epochs = 0
    vocab_size = len(tokenizer)
    padding_index = tokenizer.pad_token_id
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_training_loss = train_epoch(
                model,
                vocab_size,
                train_loader,
                optimizer,
                scheduler,
                ACCUMULATION_STEPS_DIFFUSION,
                noise_schedule,
                max_timesteps,
                padding_index,
                device,
            )
            epoch_valid_loss = eval_model(
                model,
                vocab_size,
                valid_loader,
                noise_schedule,
                max_timesteps,
                padding_index,
                device,
            )
            print(
                f"Epoch {epoch} | Training loss: {epoch_training_loss} | Valid loss: {epoch_valid_loss}",
                flush=True,
            )

            training_losses.append(epoch_training_loss)
            valid_losses.append(epoch_valid_loss)
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                no_improvement_epochs = 0
                if test_dataset:
                    save_model_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        epoch_training_loss,
                        best_valid_loss,
                        DIFFUSION_FILE_PATH,
                        "BestModel.pt",
                    )
                    torch.save(
                        {
                            "training_losses": training_losses,
                            "valid_losses": valid_losses,
                        },
                        DIFFUSION_FILE_PATH + "Losses.pt",
                    )
            else:
                no_improvement_epochs = no_improvement_epochs + 1
                if no_improvement_epochs >= EARLY_STOP_EPOCHS:
                    print(
                        f"No improvements spotted for {EARLY_STOP_EPOCHS} epochs. Stopping early...",
                        flush=True,
                    )
                    break
                if test_dataset:
                    save_model_checkpoint(
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        epoch_training_loss,
                        best_valid_loss,
                        DIFFUSION_FILE_PATH,
                        "Checkpoint.pt",
                    )
                    torch.save(
                        {
                            "training_losses": training_losses,
                            "valid_losses": valid_losses,
                        },
                        DIFFUSION_FILE_PATH + "Losses.pt",
                    )
            if trial is not None:
                trial.report(epoch_valid_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        if test_dataset:
            test_loss = eval_model(
                model,
                vocab_size,
                test_loader,
                noise_schedule,
                max_timesteps,
                padding_index,
                device,
            )
            print(f"Final Test Loss: {test_loss}", flush=True)
            end_time = time.time()
            total_duration = end_time - start_time
            print(
                f"Total Training Time: {total_duration / 3600:.2f} hours.", flush=True
            )
            torch.save(
                {
                    "training_losses": training_losses,
                    "valid_losses": valid_losses,
                    "test_loss": test_loss,
                    "total_duration": total_duration,
                },
                DIFFUSION_FILE_PATH + "Losses.pt",
            )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Exception thrown: {e}", file=sys.stderr, flush=True)
        if training_losses:
            save_model_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                epoch_training_loss,
                best_valid_loss,
                DIFFUSION_FILE_PATH,
                "Crash-Checkpoint.pt",
            )
            torch.save(
                {"training_losses": training_losses, "valid_losses": valid_losses},
                DIFFUSION_FILE_PATH + "Losses.pt",
            )
    finally:
        torch.cuda.empty_cache()
    # To help with hyperparameter tuning
    return best_valid_loss


def hyperparameter_tuning_objective_l1(
    trial, tokenizer, num_epochs, train_dataset, valid_dataset, device
):
    """
    Uses Optuna to tune the model's architectural hyperparameters.
    Trains a trial version of the model using a different nhead, nhead_scalar, num_layers, time_embedding_dim, dim_feedforward_scalar, and noise_schedule_type.

    Args:
        trial (Any): The trial object Optuna uses to keep track of the model as it trains.
        tokenizer (PreTrainedTokenizerFast): The tokenizer that the model is training to work with.
        num_epochs (int): The number of epochs to train the model for.
        train_dataset (WikitextDataset): The dataset of training data for the model.
        valid_dataset (WikitextDataset): The dataset of validation data for the model.
        device (DeviceLikeType): The device that is being used to train the model.

    Returns:
        The best validation loss for the trial model.
    """
    nhead = trial.suggest_categorical("nhead", [4, 8, 12])
    nhead_scalar = trial.suggest_int("nhead_scalar", 32, 64)
    num_layers = trial.suggest_int("n_layers", 4, 12)
    time_embedding_dim = trial.suggest_int("time_embedding_dim", 64, 128)
    dim_feedforward_scalar = trial.suggest_int("dim_feedforward_scalar", 2, 4)
    noise_schedule_type = trial.suggest_categorical(
        "noise_schedule_type", [LINEAR_NOISE_SCHEDULE, COSINE_NOISE_SCHEDULE]
    )

    # These two hyperparameters make up the d_model.
    # If it's too high, my training might crash!
    # Though this is numerically impossible with the ranges I've set, it's here as a failsafe "if the code changes".
    if (nhead * nhead_scalar) > 768:
        raise optuna.exceptions.TrialPruned()

    validation_loss = train_full_model(
        tokenizer,
        train_dataset,
        valid_dataset,
        None,
        device,
        num_epochs,
        lr=5e-4,
        max_timesteps=1000,
        nhead=nhead,
        nhead_scalar=nhead_scalar,
        num_layers=num_layers,
        time_embedding_dim=time_embedding_dim,
        dim_feedforward_scalar=dim_feedforward_scalar,
        noise_schedule_type=noise_schedule_type,
        warmup_pct_start=0.1,
        dropout=0.1,
        weight_decay=0.05,
        label_smoothing=0.1,
        trial=trial,
    )
    return validation_loss


def hyperparameter_tuning_objective_l2(
    trial, tokenizer, num_epochs, train_dataset, valid_dataset, device
):
    """
    Uses Optuna to tune the model's regularization-focused hyperparameters.
    Trains a trial version of the model using a different warmup_pct_start, dropout, weight_decay, and label_smoothing.

    Args:
        trial (Any): The trial object Optuna uses to keep track of the model as it trains.
        tokenizer (PreTrainedTokenizerFast): The tokenizer that the model is training to work with.
        num_epochs (int): The number of epochs to train the model for.
        train_dataset (WikitextDataset): The dataset of training data for the model.
        valid_dataset (WikitextDataset): The dataset of validation data for the model.
        device (DeviceLikeType): The device that is being used to train the model.

    Returns:
        The best validation loss for the trial model.
    """
    warmup_pct_start = trial.suggest_float("warmup_pct_start", 0.05, 0.3)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    validation_loss = train_full_model(
        tokenizer,
        train_dataset,
        valid_dataset,
        None,
        device,
        num_epochs,
        lr=5e-4,
        max_timesteps=1000,
        nhead=12,
        nhead_scalar=59,
        num_layers=5,
        time_embedding_dim=75,
        dim_feedforward_scalar=2,
        noise_schedule_type=COSINE_NOISE_SCHEDULE,
        warmup_pct_start=warmup_pct_start,
        dropout=dropout,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        trial=trial,
    )
    return validation_loss


if __name__ == "__main__":
    device = get_device()
    tokenizer = get_tokenizer()

    # Prepare the datasets
    train_dataset = WikitextDataset(
        TRAIN_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR
    )
    valid_dataset = WikitextDataset(
        VALID_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR
    )

    # Reduce the training set to a manageable level
    train_dataset = get_reduced_dataset(train_dataset, SUBSET_RATIO_OF_DATASET_TO_TRAIN)

    args = get_args()
    if args.tune:
        level = min(args.tune, 2)
        num_epochs = 3
        print(f"Testing hyperparameter combinations at level {level}...", flush=True)
        train_dataset = get_reduced_dataset(
            train_dataset, SUBSET_RATIO_OF_DATASET_TO_TUNE
        )

        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            study_name=f"Wikitext_Level_{DIFFUSION_MODE_INDICATOR}_{level}",
            storage=f"sqlite:///tuning_history_{DIFFUSION_MODE_INDICATOR}.db",
            load_if_exists=True,
            pruner=pruner,
        )
        if args.tune == 1:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l1(
                    trial, tokenizer, num_epochs, train_dataset, valid_dataset, device
                ),
                n_trials=10,
            )
        else:
            study.optimize(
                lambda trial: hyperparameter_tuning_objective_l2(
                    trial, tokenizer, num_epochs, train_dataset, valid_dataset, device
                ),
                n_trials=8,
            )
        print(f"Best Hyperparameters: {study.best_params}", flush=True)
    else:
        print("Training model...", flush=True)
        test_dataset = WikitextDataset(
            TEST_DATA_FILE_PATH, BLOCK_SIZE_DIFFUSION, mode=DIFFUSION_MODE_INDICATOR
        )
        train_full_model(
            tokenizer,
            train_dataset,
            valid_dataset,
            test_dataset,
            device,
            20,
            lr=5e-4,
            max_timesteps=1000,
            nhead=12,
            nhead_scalar=59,
            num_layers=5,
            time_embedding_dim=75,
            dim_feedforward_scalar=2,
            noise_schedule_type=COSINE_NOISE_SCHEDULE,
            warmup_pct_start=0.1,
            dropout=0.1,
            weight_decay=0.05,
            label_smoothing=0.1,
        )
