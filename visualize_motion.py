"""
MotionDiT Visualization — Generate and render 3D skeleton animations from text prompts.

Loads a trained MotionDiT checkpoint, runs full DDPM reverse diffusion,
extracts 22-joint skeleton positions, and saves the result as a .gif animation.

Usage:
  python visualize_motion.py --prompt "a person walks forward and waves"
  python visualize_motion.py --prompt "someone jumps" --num_frames 80 --checkpoint ./checkpoints/best_model.pt
"""

import os
import math
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from diffusers import DDPMScheduler

from motion_dit_model import MotionDiT


# HumanML3D 22-joint kinematic chain (from the repo's paramUtil.py)
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],      # Right leg
    [0, 1, 4, 7, 10],      # Left leg
    [0, 3, 6, 9, 12, 15],  # Spine to head
    [9, 14, 17, 19, 21],   # Right arm
    [9, 13, 16, 18, 20],   # Left arm
]

CHAIN_COLORS = ["red", "blue", "black", "red", "blue"]


# ============================================================
# Sinusoidal Timestep Encoding (must match train.py)
# ============================================================
def sinusoidal_embedding(timesteps, dim):
    """Convert integer timesteps to sinusoidal embeddings."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ============================================================
# DDPM Reverse Diffusion (Generation)
# ============================================================
@torch.no_grad()
def generate_motion(model, prompt, scheduler, num_frames=60, hidden_size=768, device="cpu"):
    """
    Full DDPM reverse diffusion: start from pure Gaussian noise,
    iteratively denoise to produce a clean motion sequence.

    Args:
        model: trained MotionDiT
        prompt: text string describing the desired motion
        scheduler: DDPMScheduler instance
        num_frames: number of frames to generate
        hidden_size: model hidden size (for sinusoidal encoding)
        device: torch device
    Returns:
        numpy array of shape (num_frames, 272)
    """
    model.eval()

    # Start from pure noise
    motion_shape = (1, num_frames, 272)
    noisy_motion = torch.randn(motion_shape, device=device)

    # Set up denoising schedule
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    print(f"Running {len(scheduler.timesteps)} denoising steps...")
    for i, t in enumerate(scheduler.timesteps):
        timestep_batch = t.unsqueeze(0).to(device)

        # Convert timestep to sinusoidal embedding (same as training)
        t_emb = sinusoidal_embedding(timestep_batch, hidden_size)

        # Predict the noise
        predicted_noise = model(noisy_motion, [prompt], t_emb)

        # Denoise one step
        noisy_motion = scheduler.step(predicted_noise, t, noisy_motion).prev_sample

        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{len(scheduler.timesteps)}")

    generated_motion = noisy_motion.squeeze(0).cpu().numpy()  # (num_frames, 272)
    return generated_motion


# ============================================================
# Extract Joint Positions from 272-dim representation
# ============================================================
def extract_joint_positions(motion_data):
    """
    Extract the 22 joint (x, y, z) positions from the motion vector.

    This dataset is the "272-dim Motion Representation" from
    Li-xingXiao/272-dim-Motion-Representation (downloaded via
    setup_humanml3d.py from lxxiao/272-dim-HumanML3D), NOT the standard
    263-dim HumanML3D layout. Per that project's documented column layout:

      [0:8]    8-dim root/global motion features
      [8:74]   22 joint positions (22 * 3 = 66)      <- what we want
      [74:206] 22 joint rotations, 6D repr (22 * 6 = 132)
      [206:272] 22 joint velocities (22 * 3 = 66)

    (8 + 66 + 132 + 66 = 272, matches exactly.)

    Verified against real training samples: bone lengths computed from
    this offset are constant across all frames (std/mean ~ 0.000 for
    every bone in KINEMATIC_CHAIN), confirming these are true rigid
    skeleton joint positions. The previous 4:70 slice (correct for
    standard 263-dim HumanML3D, wrong here) produced bones that visibly
    stretched/shrank by up to 10% frame-to-frame.

    Returns: (num_frames, 22, 3)
    """
    num_frames = motion_data.shape[0]
    joint_positions_flat = motion_data[:, 8:74]  # (num_frames, 66)
    joint_positions = joint_positions_flat.reshape(num_frames, 22, 3)
    return joint_positions


# Modern dark palette. One accent color per kinematic chain — no more
# flat red/blue/black, and joints get a matching gradient instead of
# solid black dots.
BG_COLOR = "#0b0f19"
FLOOR_COLOR = (0.35, 0.55, 0.95, 0.10)
GRID_COLOR = (1, 1, 1, 0.06)
TRAIL_COLOR = "#4fd1ff"
CHAIN_STYLE = [
    {"color": "#ff5c7a", "width": 4.2},  # right leg
    {"color": "#4fd1ff", "width": 4.2},  # left leg
    {"color": "#f5f5f5", "width": 5.0},  # spine -> head
    {"color": "#ff9a5c", "width": 3.2},  # right arm
    {"color": "#5cffb1", "width": 3.2},  # left arm
]


# ============================================================
# 3D Skeleton Animation (adapted from HumanML3D repo animation.ipynb)
# ============================================================
def plot_3d_motion(
    save_path,
    joints,
    title,
    fps=20,
    figsize=(9, 9),
    dpi=150,
    up_axis=1,
    elev=20,
    azim=-60,
    trail_len=15,
):
    """
    Render a 3D skeleton animation and save as .gif.

    Args:
        save_path: output .gif path
        joints: numpy array (num_frames, 22, 3)
        title: text to display as title
        fps: animation frames per second
        figsize / dpi: figure size and resolution
        up_axis: which of the 3 columns (0/1/2) is the vertical axis in
            your data. HumanML3D's own convention is 1, but if your
            custom 272-dim layout stores joints differently, the pose
            will look flipped/squashed until this matches your data —
            try 0 or 2 if the skeleton still looks wrong after this update.
        elev / azim: camera angle in degrees. The old fixed elev=120
            was tuned to flip HumanML3D's convention right-side up; with
            a configurable up_axis we use a normal, readable camera angle
            instead and you can nudge it here if needed.
        trail_len: how many past frames the fading motion trail shows.
    """
    # Wrap long titles
    title_words = title.split(" ")
    if len(title_words) > 10:
        title = "\n".join([" ".join(title_words[:10]), " ".join(title_words[10:])])

    data = joints.copy()

    # Reorder axes so index 1 is always "up" internally, regardless of
    # which raw column up_axis points to.
    if up_axis != 1:
        order = [0, 1, 2]
        order[1], order[up_axis] = order[up_axis], order[1]
        data = data[..., order]

    frame_number = data.shape[0]

    # Adjust height so the skeleton stands on the floor (y == 0 at lowest point)
    MINS = data.min(axis=0).min(axis=0)
    data[:, :, 1] -= MINS[1]

    # Recalculate bounds after the floor adjustment
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    # Center on root joint trajectory (root = joint 0)
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    # --- Autoscale the viewing volume to the actual motion extent ---
    # A fixed radius made small motions render as a tiny speck. Instead,
    # size the box around the real spread of the data plus headroom, with
    # a sane floor so a nearly-static pose doesn't zoom in absurdly close.
    span = max((MAXS - MINS).max(), 0.5)
    radius = span * 1.6
    height_top = max(MAXS[1] - MINS[1], 0.5) * 1.5

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COLOR)

    def style_axes():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, height_top])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, height_top / radius, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(BG_COLOR)
            axis.line.set_color(GRID_COLOR)
            axis._axinfo["grid"]["color"] = GRID_COLOR
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.axis("off")

    def plot_floor(index):
        verts = [
            [MINS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1]],
            [MINS[0] - trajec[index, 0], 0, MAXS[2] - trajec[index, 1]],
            [MAXS[0] - trajec[index, 0], 0, MAXS[2] - trajec[index, 1]],
            [MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1]],
        ]
        floor = Poly3DCollection([verts])
        floor.set_facecolor(FLOOR_COLOR)
        floor.set_edgecolor((1, 1, 1, 0.08))
        ax.add_collection3d(floor)

    def update(index):
        ax.cla()
        style_axes()
        fig.suptitle(title, fontsize=15, color="#f5f5f5", y=0.96)

        plot_floor(index)

        # Fading trajectory trail — recent frames are brighter/thicker,
        # older frames fade out, instead of one flat solid line.
        start = max(0, index - trail_len)
        for j in range(start, index):
            alpha = (j - start + 1) / max(1, index - start)
            ax.plot3D(
                trajec[j:j + 2, 0] - trajec[index, 0],
                [0, 0],
                trajec[j:j + 2, 1] - trajec[index, 1],
                linewidth=1.5,
                color=TRAIL_COLOR,
                alpha=alpha * 0.8,
            )

        # Skeleton bones, one accent color per limb chain
        for chain, style in zip(KINEMATIC_CHAIN, CHAIN_STYLE):
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=style["width"],
                color=style["color"],
                solid_capstyle="round",
            )

        # Joints on top, small glow via layered scatter
        ax.scatter(
            data[index, :, 0],
            data[index, :, 1],
            data[index, :, 2],
            color="white",
            edgecolor="#222",
            linewidth=0.4,
            s=28,
            depthshade=True,
            zorder=5,
        )

        ax.text2D(
            0.02, 0.02, f"frame {index + 1}/{frame_number}",
            transform=ax.transAxes, color="#888", fontsize=9,
        )

    ani = FuncAnimation(fig, update, frames=frame_number, interval=5000 / fps, repeat=False)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"Animation saved to: {save_path}")
# ============================================================
# Main
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load Checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    saved_args = checkpoint.get("args", {})
    hidden_size = saved_args.get("hidden_size", args.hidden_size)

    # ---- Build Model ----
    model = MotionDiT(
        hidden_size=hidden_size,
        num_layers=saved_args.get("num_layers", args.num_layers),
        max_frames=saved_args.get("max_frames", args.num_frames),
        alpha_init_value=saved_args.get("alpha_init", 0.5),
        shift_init_value=saved_args.get("shift_init", 0.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Model loaded (trained for {checkpoint.get('epoch', '?')} epochs, "
        f"loss: {checkpoint.get('loss', 0):.6f})"
    )

    # ---- Noise Scheduler ----
    scheduler = DDPMScheduler(
        num_train_timesteps=saved_args.get("num_timesteps", 1000),
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
    )

    # ---- Generate Motion ----
    print(f'\nGenerating motion for prompt: "{args.prompt}"')
    print(f"Frames: {args.num_frames}")

    generated_motion = generate_motion(
        model, args.prompt, scheduler,
        num_frames=args.num_frames,
        hidden_size=hidden_size,
        device=device,
    )
    print(f"Generated motion shape: {generated_motion.shape}")

    # ---- Save Raw .npy ----
    os.makedirs(args.output_dir, exist_ok=True)
    npy_path = os.path.join(args.output_dir, "generated_motion.npy")
    np.save(npy_path, generated_motion)
    print(f"Raw motion saved to: {npy_path}")

    # ---- Extract Joints and Animate ----
    joints = extract_joint_positions(generated_motion)
    print(f"Extracted joints shape: {joints.shape}")

    gif_path = os.path.join(args.output_dir, "generated_motion.gif")
    plot_3d_motion(
        gif_path, joints, title=args.prompt, fps=args.fps,
        up_axis=args.up_axis, elev=args.elev, azim=args.azim,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize motion from text")

    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for motion generation")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--num_frames", type=int, default=100,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=8,
                        help="Animation FPS")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--up_axis", type=int, default=1, choices=[0, 1, 2],
                        help="Which raw joint column (x=0,y=1,z=2) is vertical "
                             "in your data. Try 0 or 2 if the pose looks "
                             "flipped/squashed with the default.")
    parser.add_argument("--elev", type=float, default=20.0,
                        help="Camera elevation angle (degrees)")
    parser.add_argument("--azim", type=float, default=-45.0,
                        help="Camera azimuth angle (degrees)")

    # Model defaults (used if checkpoint doesn't contain args)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)

    args = parser.parse_args()
    main(args)
