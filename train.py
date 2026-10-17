"""
Training pipeline for MotionDiT — Text-to-Motion Diffusion Model.

Features:
  - DDPM noise scheduling via HuggingFace diffusers
  - Cosine LR scheduler with linear warmup
  - Early stopping with configurable patience
  - TensorBoard logging (per-step loss, LR, epoch averages)
  - Masked MSE loss (ignores padded frames)
  - Gradient clipping
  - Best model + periodic checkpointing

Usage:
  python train.py --data_dir ./data
  python train.py --data_dir ./data --batch_size 32 --epochs 200 --hidden_size 768
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler

from motion_dit_model import MotionDiT
from humanml3d_dataset import get_dataloader


# ============================================================
# Early Stopping
# ============================================================
class EarlyStopping:
    """Stops training when loss stops improving for `patience` epochs."""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ============================================================
# Sinusoidal Timestep Encoding (to feed into diffusers TimestepEmbedding)
# ============================================================
def sinusoidal_embedding(timesteps, dim):
    """
    Convert integer timesteps to sinusoidal embeddings.
    The diffusers TimestepEmbedding MLP expects pre-encoded vectors.

    Args:
        timesteps: (B,) integer tensor
        dim: embedding dimension (must match model hidden_size)
    Returns:
        (B, dim) sinusoidal embedding
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ============================================================
# Training Function
# ============================================================
def train(args):
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Directories ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- DataLoader ----
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    model = MotionDiT(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_frames=args.max_frames,
        alpha_init_value=args.alpha_init,
        shift_init_value=args.shift_init,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---- Noise Scheduler ----
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_timesteps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---- LR Scheduler (Cosine Annealing with Linear Warmup) ----
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Early Stopping ----
    early_stopping = EarlyStopping(patience=args.patience)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']}, loss {best_loss:.6f}")

    # ---- Training Loop ----
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Warmup steps: {warmup_steps}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            clean_skeleton = batch["skeleton"].to(device)  # (B, F, 272)
            texts = batch["text"]                           # list[str]
            valid_lens = batch["valid_len"].to(device)      # (B,)
            batch_size = clean_skeleton.shape[0]

            # 1. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device,
            ).long()

            # 2. Sample noise
            noise = torch.randn_like(clean_skeleton)

            # 3. Add noise to skeleton using the scheduler
            noisy_skeleton = noise_scheduler.add_noise(clean_skeleton, noise, timesteps)

            # 4. Convert integer timesteps to sinusoidal embeddings
            #    (diffusers TimestepEmbedding expects pre-encoded vectors)
            t_emb = sinusoidal_embedding(timesteps, args.hidden_size)

            # 5. Predict the noise
            predicted_noise = model(noisy_skeleton, texts, t_emb, valid_lens=valid_lens)

            # 6. Masked MSE loss (ignore padded frames)
            mask = torch.arange(args.max_frames, device=device).unsqueeze(0) < valid_lens.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (B, F, 1)

            loss = ((predicted_noise - noise) ** 2 * mask).sum() / (
                mask.sum() * noise.shape[-1]
            ).clamp_min(1.0)

            # 7. Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # ---- TensorBoard Logging ----
            writer.add_scalar("Train/loss_step", loss.item(), global_step)
            writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            if (batch_idx + 1) % args.log_every == 0:
                print(
                    f"  Epoch [{epoch}/{args.epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        # ---- Epoch Summary ----
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        writer.add_scalar("Train/loss_epoch", avg_loss, epoch)

        print(
            f"\nEpoch [{epoch}/{args.epochs}] "
            f"Avg Loss: {avg_loss:.6f} "
            f"Time: {epoch_time:.1f}s"
        )

        # ---- Save Best Model ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"  -> Saved best model (loss: {best_loss:.6f})")

        # ---- Periodic Checkpoint ----
        if epoch % args.save_every == 0:
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"  -> Saved checkpoint at epoch {epoch}")

        # ---- Early Stopping ----
        if early_stopping(avg_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

        print("-" * 60)

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f'Best model saved to: {os.path.join(args.checkpoint_dir, "best_model.pt")}')
    print(f"TensorBoard logs at: {args.log_dir}")
    print(f"  -> Run: tensorboard --logdir {args.log_dir}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MotionDiT")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to data directory containing poses/ and texts/")
    parser.add_argument("--max_frames", type=int, default=120,
                        help="Maximum sequence length (frames)")
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Transformer hidden size")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of DiT blocks")
    parser.add_argument("--alpha_init", type=float, default=0.5,
                        help="Dynamic_erf alpha init value")
    parser.add_argument("--shift_init", type=float, default=0.0,
                        help="Dynamic_erf shift init value")

    # Training
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    # Logging & Checkpoints
    parser.add_argument("--log_dir", type=str, default="./runs/motion_dit",
                        help="TensorBoard log directory")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Print log every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--save_every", type=int, default=25,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()
    train(args)
