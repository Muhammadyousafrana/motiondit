"""MotionDiT: a text-conditioned diffusion transformer for 272-D motion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.embeddings import TimestepEmbedding

# ============================================================
# Conditioning Block (CLIP Text + Timestep Embedding)
# ============================================================
class ConditioningBlock(nn.Module):
    def __init__(self, hidden_size=768):
        super(ConditioningBlock, self).__init__()
        model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        self.text_tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.timestep_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, hidden_size)

    def train(self, mode=True):
        """Keep the frozen CLIP encoder in eval mode when the parent trains."""
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, text, timestep):
        device = next(self.text_encoder.parameters()).device
        t_emb = self.timestep_embedding(timestep)
        text_input = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_input = {k: v.to(device) for k, v in text_input.items()}
        with torch.no_grad():
            text_outputs = self.text_encoder(**text_input)
        text_emb = text_outputs.pooler_output if hasattr(text_outputs, 'pooler_output') else text_outputs.last_hidden_state[:, 0]
        text_emb = self.text_projector(text_emb)
        conditioning = text_emb + t_emb
        return conditioning


# ============================================================
# Noise Patch Block (Frame Embedding + Positional Encoding)
# ============================================================
class NoisePatchBlock(nn.Module):
    def __init__(self, hidden_size=768, max_frames=200):
        super(NoisePatchBlock, self).__init__()
        self.token_embedder = nn.Linear(272, hidden_size)
        self.pos_embedder = nn.Parameter(torch.randn(1, max_frames, hidden_size))

    def forward(self, noisy_skelton):
        x = self.token_embedder(noisy_skelton)
        frames = x.shape[1]
        if frames > self.pos_embedder.shape[1]:
            raise ValueError(
                f"Input has {frames} frames, exceeding configured max_frames="
                f"{self.pos_embedder.shape[1]}"
            )
        pos_emb = self.pos_embedder[:, :frames, :]
        x = x + pos_emb
        return x


# ============================================================
# Dynamic Erf Normalization (LayerNorm followed by a learned erf transform)
# ============================================================
class Dynamic_erf(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        # Normalize first: erf(alpha*x + shift) alone is an activation, not a
        # LayerNorm replacement, and can saturate as residual activations grow.
        x = F.layer_norm(x, self.normalized_shape)
        x = self.alpha * x + self.shift
        if self.elementwise_affine:
            return torch.erf(x) * self.weight + self.bias
        else:
            return torch.erf(x)

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, "
                f"elementwise_affine={self.elementwise_affine}, "
                f"alpha_init_value={self.alpha_init_value}, "
                f"shift_init_value={self.shift_init_value}")


def convert_ln_to_derf(module, alpha_init_value=0.5, shift_init_value=0.0):
    """Recursively replace all LayerNorm layers with Dynamic_erf."""
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = Dynamic_erf(
            module.normalized_shape, module.elementwise_affine,
            alpha_init_value=alpha_init_value, shift_init_value=shift_init_value
        )
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_ln_to_derf(child, alpha_init_value, shift_init_value)
        )
    del module
    return module_output


# ============================================================
# Attention Block (one self-attention operation per DiT block)
# ============================================================
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=768):
        super(AttentionBlock, self).__init__()
        if hidden_size % 12 != 0:
            raise ValueError("hidden_size must be divisible by 12 (the attention head count)")
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=12, dropout=0.1, batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        out, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return out


# ============================================================
# Pointwise Feed-Forward Network
# ============================================================
class PointwiseFeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


# ============================================================
# DiT Block (pre-normalization + Attention + Feed-Forward with residuals)
# ============================================================
class DiTBlock(nn.Module):
    def __init__(self, hidden_size=768, alpha_init_value=0.5, shift_init_value=0.0):
        super(DiTBlock, self).__init__()
        self.ln_1 = Dynamic_erf(hidden_size, True, alpha_init_value, shift_init_value)
        self.ln_2 = Dynamic_erf(hidden_size, True, alpha_init_value, shift_init_value)
        self.attention = AttentionBlock(hidden_size)
        self.point_wise_ff = PointwiseFeedForward(hidden_size, hidden_size * 4, 0.1)

    def forward(self, x, key_padding_mask=None):
        x_add = x + self.attention(self.ln_1(x), key_padding_mask)
        x = x_add + self.point_wise_ff(self.ln_2(x_add))
        return x


# ============================================================
# Full Model
# ============================================================
class MotionDiT(nn.Module):
    def __init__(self, hidden_size=768, num_layers=6, max_frames=200,
                 alpha_init_value=0.5, shift_init_value=0.0):
        super(MotionDiT, self).__init__()
        self.motion_features = 272
        self.conditioning_block = ConditioningBlock(hidden_size)
        self.noise_patch_block = NoisePatchBlock(hidden_size, max_frames=max_frames)
        self.max_frames = max_frames
        self.dit_blocks = nn.Sequential(
            *[DiTBlock(hidden_size, alpha_init_value, shift_init_value) for _ in range(num_layers)]
        )
        self.layer_norm = Dynamic_erf(hidden_size, True, alpha_init_value, shift_init_value)
        self.linear1 = nn.Linear(hidden_size, 512)
        self.output = nn.Linear(512, 272)

    def forward(self, noise_skeleton, text, timestep, valid_lens=None):
        cond_x = self.conditioning_block(text, timestep)
        noise_sk_x = self.noise_patch_block(noise_skeleton)
        key_padding_mask = None
        if valid_lens is not None:
            if valid_lens.shape[0] != noise_skeleton.shape[0]:
                raise ValueError("valid_lens must contain one length per batch item")
            frame_padding = (
                torch.arange(noise_skeleton.shape[1], device=noise_skeleton.device)[None, :]
                >= valid_lens.to(noise_skeleton.device)[:, None]
            )
            # The leading text/timestep conditioning token is always valid.
            key_padding_mask = torch.cat([
                torch.zeros((noise_skeleton.shape[0], 1), dtype=torch.bool,
                            device=noise_skeleton.device),
                frame_padding,
            ], dim=1)
        # Prepend conditioning token to the sequence
        x = torch.cat([cond_x.unsqueeze(1), noise_sk_x], dim=1)
        for block in self.dit_blocks:
            x = block(x, key_padding_mask)
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.output(x)
        # Remove the conditioning token from output (first token)
        x = x[:, 1:, :]
        return x


# Keep existing training and generation scripts importing the old class name.
MyDiTransformer = MotionDiT


# ============================================================
# Quick Smoke Test
# ============================================================
if __name__ == "__main__":
    print("Building MyDiTransformer...")
    model = MyDiTransformer(hidden_size=768, num_layers=2)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    B, F = 2, 60
    noisy_skeleton = torch.randn(B, F, 272)
    timesteps = torch.randn(B, 768)  # TimestepEmbedding from diffusers expects pre-embedded
    texts = ["a person walks forward", "someone jumps up"]

    predicted_noise = model(noisy_skeleton, texts, timesteps)
    print(f"\nInput shape:  {noisy_skeleton.shape}")
    print(f"Output shape: {predicted_noise.shape}")
    print("Smoke test passed!")
