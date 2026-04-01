from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import argparse
import logging
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.model.model import GPTLanguageModel

def setup_logger(rank, batch_size, world_size, log_file=None):
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f'training_b{batch_size}_g{world_size}_{timestamp}.log'

    logger = logging.getLogger(f'rank_{rank}')
    logger.setLevel(logging.INFO)

    # File handler (shared between ranks)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Format with rank information
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)s - %(message)s')

    # Add rank to the log record
    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    logger.addFilter(RankFilter())
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (only for rank 0)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model with distributed training')
    parser.add_argument('--num_gpus', type=int, default=-1,
                       help='Number of GPUs to use. -1 means use all available GPUs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Global batch size')
    parser.add_argument('--epochs', type=int, default=1,  # Changed from max_iters to epochs
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=500,
                       help='How often to log batch loss to wandb')
    return parser.parse_args()


def setup(rank, world_size):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        raise

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception as e:
        print(f"Failed to cleanup process group: {e}")

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

def train(rank, world_size, args):
    # Setup logger
    logger = setup_logger(
            rank=rank,
            batch_size=args.batch_size,
            world_size=world_size
        )

    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Training Hyperparameters
    batch_size = args.batch_size // world_size  # Adjust batch size per GPU
    block_size = 256
    epochs = args.epochs
    learning_rate = args.learning_rate
    log_interval = args.log_interval
    eval_batches = 300
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project="gpt-distributed",
            config={
                "batch_size": batch_size * world_size,
                "block_size": block_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "eval_batches": eval_batches,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
                "log_interval": log_interval,
            }
        )

    # Load and preprocess data
    input_path = PROJECT_ROOT / "transformer" / "input.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Create datasets
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Create Data Loaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
    )

    # Create model and move to GPU
    model = GPTLanguageModel(vocab_size)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        logger.info("Starting loss estimation on rank 0")
        out = {}
        model.eval()

        try:
            for split, loader in [('train', train_loader), ('val', val_loader)]:
                logger.info(f"Evaluating {split} split")
                losses = []
                for batch_idx, (x, y) in enumerate(loader):
                    if batch_idx >= eval_batches:  # Stop after eval_batches batches
                         logger.info(f"Reached {eval_batches} batches, stopping evaluation")
                         break
                    logger.info(f"Eval batch {batch_idx}")
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    losses.append(loss.item())
                    logger.info(f"Batch {batch_idx} loss: {loss.item():.4f}")

                avg_loss = sum(losses) / len(losses)
                logger.info(f"Average {split} loss: {avg_loss:.4f}")
                out[split] = avg_loss

        except Exception as e:
            logger.error(f"Error in loss estimation: {str(e)}")
            raise
        finally:
            model.train()

        return out

    # Training loop
    logger.info("Starting training process")

    total_batches = len(train_loader)
    logger.info(f"Rank {rank}: Total batches per epoch: {total_batches}")

    # Global batch counter for continuous batch logging across epochs
    global_batch = 0

    for epoch in range(epochs):
        logger.info(f"Rank {rank}: Starting epoch {epoch}")
        train_sampler.set_epoch(epoch)  # Important for proper shuffling

        # Training phase
        model.train()
        total_loss = 0

        # Initialize timing counters
        data_loading_time = 0.0
        data_movement_time = 0.0
        training_compute_time = 0.0
        training_communication_time = 0.0

        # Start timing for data loading
        if device.type == "cuda":
            torch.cuda.synchronize()
        data_loading_start_time = time.perf_counter()

        for batch_idx, (x, y) in enumerate(train_loader):
            # End timing for data loading
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_loading_end_time = time.perf_counter()
            data_loading_time += data_loading_end_time - data_loading_start_time

            # Start timing for data movement
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_movement_start_time = time.perf_counter()

            x, y = x.to(device), y.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            data_movement_end_time = time.perf_counter()
            data_movement_time += data_movement_end_time - data_movement_start_time

            # Start timing for training compute
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_compute_start = time.perf_counter()

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            training_compute_end = time.perf_counter()
            training_compute_time += training_compute_end - training_compute_start

            # Start timing for training communication/gradient sync
            if device.type == "cuda":
                torch.cuda.synchronize()
            comm_start = time.perf_counter()

            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            comm_end = time.perf_counter()
            training_communication_time += comm_end - comm_start

            total_loss += loss.item()

            # Log batch loss from all ranks
            if batch_idx % log_interval == 0:
                logger.info(f'Rank {rank}: Epoch: {epoch}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.4f}')
                # Log to wandb with rank information
                if rank == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": global_batch,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx
                    })

            global_batch += 1
            # Reset data loading timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_loading_start_time = time.perf_counter()

        # Calculate average loss for the epoch
        avg_loss = total_loss / total_batches

        # Total epoch time
        total_epoch_time = data_loading_time + data_movement_time + training_compute_time + training_communication_time

        # Log timing information
        logger.info(f"""
                Rank {rank} Epoch {epoch} timing breakdown:
                    Average Loss: {avg_loss:.2f}
                    Total epoch time: {total_epoch_time:.2f} seconds
                    Data loading time: {data_loading_time:.2f} seconds
                    Data movement time: {data_movement_time:.2f} seconds
                    Training compute time: {training_compute_time:.2f} seconds
                    Training communication time: {training_communication_time:.2f} seconds
                """)

        # Evaluation phase
        if rank == 0:
            losses = estimate_loss()
            logger.info(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Log metrics for this rank only
            wandb.log({
                "epoch": epoch,
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "training_loss": avg_loss,
                "epoch_completed": epoch + 1,
                'epoch_total_time': total_epoch_time,
                'epoch_data_loading_time': data_loading_time,
                'epoch_data_movement_time': data_movement_time,
                'epoch_compute_time': training_compute_time,
                'epoch_communication_time': training_communication_time,
            })

        # Synchronize before starting next epoch
        dist.barrier()

    # Save model and generate text only on rank 0
    if rank == 0:
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', 'final_model.pth')
        torch.save(model.module.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path, base_path=os.getcwd())

        vocab_path = os.path.join('checkpoints', 'vocab_info.pth')
        torch.save({'stoi': stoi, 'itos': itos}, vocab_path)
        wandb.save(vocab_path, base_path=os.getcwd())

        # Generate sample text
        context = torch.zeros((1, 1), dtype=torch.long, device=rank)
        generated_text = decode(model.module.generate(context, max_new_tokens=500)[0].tolist())
        wandb.log({"generated_text": generated_text})

        wandb.finish()

    # Clean up
    cleanup()


if __name__ == "__main__":
    args = parse_args()

    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    if args.num_gpus == -1:
        world_size = available_gpus
    else:
        world_size = min(args.num_gpus, available_gpus)
        if world_size < args.num_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available.")

    if world_size == 0:
        raise ValueError("No GPUs available for training!")

    print(f"Training with {world_size} GPUs")

    try:
        torch.multiprocessing.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        # Clean up in case of error
        if dist.is_initialized():
            cleanup()
