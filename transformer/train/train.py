from pathlib import Path
import sys

import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.model.model import GPTLanguageModel


def setup_logger(batch_size, world_size, log_file=None):
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"training_b{batch_size}_g{world_size}_{timestamp}.log"

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="How often to log batch loss to wandb",
    )
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def main():
    args = parse_args()
    logger = setup_logger(batch_size=args.batch_size, world_size=1)

    # Training Hyperparameters
    batch_size = args.batch_size
    block_size = 256
    epochs = args.epochs
    learning_rate = args.learning_rate
    log_interval = args.log_interval
    eval_batches = 300
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    # Initialize wandb
    wandb.init(
        project="gpt",
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "eval_batches": eval_batches,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "log_interval": log_interval,
        },
    )

    torch.manual_seed(1337)

    # Load data
    input_path = PROJECT_ROOT / "transformer" / "input.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Create datasets
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create model
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        logger.info("Starting loss estimation")
        out = {}
        model.eval()

        for split, loader in [("train", train_loader), ("val", val_loader)]:
            logger.info(f"Evaluating {split} split")
            losses = []
            for batch_idx, (x, y) in enumerate(loader):
                if batch_idx >= eval_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            out[split] = avg_loss
            logger.info(f"Average {split} loss: {avg_loss:.4f}")

        model.train()
        return out

    # Training loop
    logger.info("Starting training process")

    total_batches = len(train_loader)
    logger.info(f"Total batches per epoch: {total_batches}")

    # Global batch counter for continuous batch logging across epochs
    global_batch = 0

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}")

        # Training phase
        model.train()
        total_loss = 0

        data_loading_time = 0.0
        data_movement_time = 0.0
        training_compute_time = 0.0
        training_communication_time = 0.0

        # Start timing for data loading
        # if device.type == "cuda":
        #     torch.cuda.synchronize()
        data_loading_start_time = time.perf_counter()

        for batch_idx, (x, y) in enumerate(train_loader):

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            data_loading_end_time = time.perf_counter()
            data_loading_time += data_loading_end_time - data_loading_start_time

            # Start timing for data movement
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            data_movement_start_time = time.perf_counter()

            x, y = x.to(device), y.to(device)

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            data_movement_end_time = time.perf_counter()
            data_movement_time += data_movement_end_time - data_movement_start_time

            # Start timing for training compute
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            training_compute_start = time.perf_counter()

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            training_compute_end = time.perf_counter()
            compute_time = training_compute_end - training_compute_start

            # Start timing for training communication/ gradient sync
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            comm_start = time.perf_counter()

            optimizer.step()

            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            comm_end = time.perf_counter()
            communication_time = comm_end - comm_start

            training_compute_time += compute_time
            training_communication_time += communication_time

            total_loss += loss.item()

            # Log batch loss every log_interval batches
            if global_batch % log_interval == 0:
                logger.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.4f}"
                )
                wandb.log(
                    {
                        "batch": global_batch,
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx,
                    }
                )

            global_batch += 1

            # Reset data loading timer
            # if device.type == "cuda":
            #     torch.cuda.synchronize()
            data_loading_start_time = time.perf_counter()

        # Calculate average loss for the epoch
        avg_loss = total_loss / total_batches

        # Total epoch time
        total_epoch_time = (
            data_loading_time
            + data_movement_time
            + training_compute_time
            + training_communication_time
        )

        # Log detailed timing information
        logger.info(
            f"""
        Epoch {epoch} timing breakdown:
            Total epoch time: {total_epoch_time:.2f} seconds
            Data loading time: {data_loading_time:.2f} seconds
            Data movement time: {data_movement_time:.2f} seconds
            Training compute time: {training_compute_time:.2f} seconds
            Training communication time: {training_communication_time:.2f} seconds
        """
        )

        # Evaluation phase
        losses = estimate_loss()
        logger.info(
            f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

        # Log metrics
        # Log metrics including timing information
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "avg_training_loss": avg_loss,
                "epoch_completed": epoch + 1,
                "total_epoch_time": total_epoch_time,
                "data_loading_time": data_loading_time,
                "data_movement_time": data_movement_time,
                "training_compute_time": training_compute_time,
                "training_communication_time": training_communication_time,
            }
        )

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "final_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path, base_path=os.getcwd())

    vocab_path = os.path.join("checkpoints", "vocab_info.pth")
    torch.save({"stoi": stoi, "itos": itos}, vocab_path)
    wandb.save(vocab_path, base_path=os.getcwd())

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    wandb.log({"generated_text": generated_text})

    wandb.finish()


if __name__ == "__main__":
    main()
