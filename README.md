# MotionDiT: Text-to-Motion Diffusion Transformer

<p align="center">
  <img src="assets/motiondit-banner.svg" alt="MotionDiT: text-conditioned human motion generation with a diffusion transformer" width="100%" />
</p>

MotionDiT is a PyTorch research and learning project for generating human motion from text. It represents each frame as a 272-dimensional motion vector, conditions a transformer on a frozen CLIP text encoder and diffusion timestep, and trains the model to predict the Gaussian noise added to motion sequences.

The project includes data preparation, a variable-length motion dataset, DDPM training, checkpoint resume, TensorBoard metrics, and a script that generates and renders 3D motion as a GIF.

> This is a custom implementation and training setup. It is not a reproduction of the original DiT paper or a published HumanML3D benchmark. No quantitative quality claim is made here.

## Architecture

For a batch of padded motion sequences `x ∈ R^(B×F×272)`, the model:

1. Projects each noisy frame into the transformer hidden dimension and adds a learned positional embedding.
2. Encodes the text with a frozen CLIP ViT-L/14 text encoder, projects the text embedding, and adds a sinusoidal diffusion-timestep embedding.
3. Prepends the combined text/timestep vector as a conditioning token.
4. Applies `num_layers` pre-normalized transformer blocks. Each block has one multi-head self-attention operation and one 4× GELU feed-forward network.
5. Projects frame tokens back to 272 dimensions as an epsilon (noise) prediction; the conditioning token is removed from the output.

The default configuration uses a hidden size of 768, 12 attention heads, 6 transformer blocks, 120 motion frames, and 1,000 DDPM training steps. The custom `Dynamic_erf` normalization applies layer normalization followed by a learned scaled and shifted error-function transform. The default loss ignores padded frames and divides by the number of valid frame features.

Padded frames are passed as an attention key-padding mask, so valid frames cannot attend to noisy padding. Text conditioning is concatenated as one token; the model does not use DiT-style adaptive LayerNorm modulation.

### 272-D motion representation

The visualization code expects the dataset's custom 272-D format (not the common 263-D HumanML3D layout):

| Feature columns | Meaning |
|---|---|
| `0:8` | Root/global motion features |
| `8:74` | 22 joint positions (3 values each) |
| `74:206` | 22 joint rotations (6D representation) |
| `206:272` | 22 joint velocities (3 values each) |

The dataset loader crops sequences longer than `max_frames` and zero-pads shorter ones. It randomly selects one text annotation for each sample when loading it.

## Requirements and installation

- Python 3.10–3.12
- The CUDA 12.6 PyTorch build pinned by the supplied requirements, or a compatible PyTorch/torchvision pair for your platform
- A CUDA-capable GPU is strongly recommended. This model uses a large frozen text encoder as well as a transformer.

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

The first model construction downloads the configured CLIP weights from Hugging Face and may require network access and several gigabytes of cache space. The supplied dependency list is a machine-specific CUDA lock; choose a matching PyTorch wheel if your hardware or Python version differs.

## Prepare the data

The setup script downloads the 272-D pose arrays and text annotations from the configured Hugging Face dataset and extracts them into `data/poses/` and `data/texts/`:

```bash
python setup_humanml3d.py
```

To choose a different destination, call the helper from Python:

```python
from setup_humanml3d import prepare_humanml3d_data

prepare_humanml3d_data(output_dir="./data")
```

Each pose file should be a NumPy array shaped `[frames, 272]`; the matching text file should contain one or more annotation lines. Training expects matching file IDs in both folders.

## Train

```bash
python train.py --data_dir ./data --batch_size 8 --epochs 200
```

Useful options:

```bash
python train.py --help
python train.py --data_dir ./data --max_frames 120 --hidden_size 768 --num_layers 6
python train.py --data_dir ./data --resume ./checkpoints/checkpoint_epoch_25.pt
```

The script uses DDPM epsilon prediction, AdamW, gradient clipping, cosine decay with warmup, and early stopping. Reduce `--batch_size` if GPU memory is limited. The current script has a single shuffled training loader and uses training loss for early stopping; it does not create a held-out validation set.

The best checkpoint is written to `checkpoints/best_model.pt`; periodic checkpoints are written every 25 epochs by default. Checkpoints contain model and optimizer state, scheduler state, epoch, loss, and the training arguments. TensorBoard event files are written under `runs/motion_dit/`:

```bash
tensorboard --logdir runs/motion_dit
```

## Generate and visualize motion

After training, run reverse diffusion from Gaussian noise and save a NumPy sequence plus a 3D GIF:

```bash
python visualize_motion.py --prompt "a person walks forward and waves" --checkpoint ./checkpoints/best_model.pt --num_frames 100
```

By default, outputs are saved under `outputs/` as `generated_motion.npy` and `generated_motion.gif`. The renderer exposes `--up_axis`, `--elev`, and `--azim` options for adjusting the camera and vertical axis.

Generation uses the training checkpoint's model depth, hidden size, and maximum frame length when these arguments are present in the checkpoint. Requested generation length must not exceed the trained positional-embedding length.

## Project files

```text
.
├── motion_dit_model.py    # Conditioning, motion embeddings, and transformer
├── humanml3d_dataset.py   # Pose/text pairing, padding, and DataLoader
├── setup_humanml3d.py     # Dataset download and extraction
├── train.py               # DDPM training and checkpointing
├── visualize_motion.py    # Reverse diffusion and 3D GIF rendering
├── text_process.py        # Optional caption token/part-of-speech processing
├── requirements.txt       # Pinned environment dependencies
├── assets/                # README banner
└── data/                  # Created by the data setup script
```

`text_process.py` is an optional caption preprocessing utility. It needs `pandas`, `spacy`, and the `en_core_web_sm` spaCy model, which are not included in the pinned training requirements:

```bash
python -m pip install pandas spacy
python -m spacy download en_core_web_sm
```

## Architecture review notes

- The old attention module nested a six-layer `TransformerEncoder` inside every outer DiT block. With six outer blocks, the default therefore built 36 self-attention layers and also duplicated feed-forward networks. It now has one self-attention and one feed-forward sublayer per outer block.
- The previous `Dynamic_erf` did not normalize its input despite replacing `LayerNorm`. It now normalizes before applying the learned erf transform.
- Padding was omitted from attention even though the loss masked padded frames. The model now masks padded positions as keys, and the loss denominator counts valid frame features.
- These architecture corrections change model state-dict keys and shapes. Checkpoints made with the earlier model cannot be loaded strictly into the corrected architecture; retrain before using the updated generator.

## Reference

Peebles, W. and Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). This project borrows the broad diffusion-transformer idea and uses its own conditioning, normalization, and motion representation.
