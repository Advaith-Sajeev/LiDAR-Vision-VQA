# deepencoder/clip_sdpa.py
# Compatible with Torch 1.13 (adds SDP fallback) and optional flash-attn
# Exposes CLIP module names that are convenient LoRA targets.
# ------------- LoRA targeting helpers (optional) -------------
def clip_l_lora_default_targets() -> Tuple[str, ...]:
    """
    Return a default list of Linear submodule names commonly targeted by LoRA.
    These match this file's module attribute names.
    """
    # Attention: qkv and output proj
    attn = ("qkv_proj", "out_proj")
    # MLP: first/second projection
    mlp  = ("mlp.fc1", "mlp.fc2")
    # LayerNorm is typically not LoRA'd, but exposed here for completeness:
    # ln   = ("layer_norm1", "layer_norm2")
    return (*attn, *mlp)

__all__ = ["VitModel", "build_clip_l", "clip_l_lora_default_targets", "vit_model_cfg"]

from contextlib import nullcontext
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict as adict

# -----------------------------
# Optional flash-attn imports
# -----------------------------
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False

    def flash_attn_qkvpacked_func(*args, **kwargs):
        raise RuntimeError("flash-attn not available")

    def flash_attn_func(*args, **kwargs):
        raise RuntimeError("flash-attn not available")


# -----------------------------
# Torch 1.13-safe SDPA fallback
# -----------------------------
_HAS_SDP = hasattr(F, "scaled_dot_product_attention")

def sdp_attention(q, k, v, attn_mask: Optional[torch.Tensor] = None):
    """
    q, k, v: [B, H, S, D]
    attn_mask: additive mask/bias [B, H, S, S] or None
    returns: [B, H, S, D]
    """
    if _HAS_SDP and attn_mask is None:
        # Native fast path (exists on torch >= 2.0; not on 1.13)
        return F.scaled_dot_product_attention(q, k, v)

    # Manual attention (Torch 1.13-compatible)
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)  # [B,H,S,S]
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# -----------------------------
# Helper(s)
# -----------------------------
@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def get_abs_pos(abs_pos: torch.Tensor, tgt_tokens: int) -> torch.Tensor:
    """
    Resample CLIP absolute position embeddings (nn.Embedding) if needed.

    abs_pos: [1, Npos, C]
    tgt_tokens: target sequence length (1 + HW)

    Returns: [1, tgt_tokens, C]
    """
    # Shape safety: we expect [1, N, C]
    if abs_pos.dim() != 3 or abs_pos.size(0) != 1:
        return abs_pos

    # Separate cls + grid
    cls = abs_pos[:, :1, :]            # [1,1,C]
    grid = abs_pos[:, 1:, :]           # [1,HW,C]
    src_hw = grid.size(1)
    if src_hw == tgt_tokens - 1:
        return abs_pos  # same length, nothing to do

    # infer src grid side and target side
    src_side = int(math.sqrt(src_hw))
    tgt_side = int(math.sqrt(max(tgt_tokens - 1, 1)))
    if src_side * src_side != src_hw or tgt_side * tgt_side != (tgt_tokens - 1):
        # Fallback: length mismatch not square; best-effort truncate/pad
        if tgt_tokens <= abs_pos.size(1):
            return abs_pos[:, :tgt_tokens, :]
        else:
            # pad with zeros
            pad = torch.zeros(1, tgt_tokens - abs_pos.size(1), abs_pos.size(2),
                              dtype=abs_pos.dtype, device=abs_pos.device)
            return torch.cat([abs_pos, pad], dim=1)

    # Resample grid via bicubic
    grid_4d = grid.transpose(1, 2).reshape(1, grid.size(2), src_side, src_side)  # [1,C,H,W]
    grid_4d = grid_4d.to(torch.float32)
    grid_4d = F.interpolate(grid_4d, size=(tgt_side, tgt_side),
                            mode="bicubic", align_corners=False, antialias=True)
    grid_4d = grid_4d.to(abs_pos.dtype)
    grid_new = grid_4d.reshape(1, grid.size(2), tgt_side * tgt_side).transpose(1, 2)  # [1,HW',C]
    return torch.cat([cls, grid_new], dim=1)  # [1,1+HW',C]


# -----------------------------
# Embeddings
# -----------------------------
class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, num_channels=3):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # Only used when patch_embeds is None (we usually bypass with SAM features)
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)),
                             persistent=False)

    def forward(self, pixel_values: torch.Tensor, patch_embeds: Optional[torch.Tensor]):
        """
        pixel_values: [B,3,H,W]
        patch_embeds: [B, C=1024, Hs, Ws] from SAM (preferred) or None to use CLIP's own patcher
        Returns: token embeddings [B, 1+HW, C]
        """
        B = pixel_values.shape[0]

        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)  # [B, C, Hs, Ws]

        tokens = patch_embeds.flatten(2).transpose(1, 2)  # [B, HW, C]
        cls_tok = self.class_embedding.expand(B, 1, -1)   # [B, 1, C]
        embeddings = torch.cat([cls_tok, tokens], dim=1)  # [B, 1+HW, C]

        # Add (resampled) absolute positions
        pos = self.position_embedding(self.position_ids)  # [1, Npos, C]
        pos = get_abs_pos(pos, embeddings.size(1))
        embeddings = embeddings + pos

        return embeddings


# -----------------------------
# FeedForward
# -----------------------------
class NoTPFeedForward(nn.Module):
    def __init__(self, cfg, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


# -----------------------------
# Attention (flash-attn optional, SDP fallback)
# -----------------------------
class NoTPAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length

        self.use_flash_attention = bool(getattr(cfg, "use_flash_attn", False) and _HAS_FLASH_ATTN)

        self.qkv_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

        self.attn_drop = cfg.attention_dropout

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, C]
        returns: [B, S, C]
        """
        B, S, C = x.shape
        qkv = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim)  # [B,S,3,H,D]

        if self.use_flash_attention:
            # flash_attn expects [B,S,3,H,D]
            out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)  # [B,S,H*D]
            out = out.view(B, S, -1)
        else:
            # Split and go through (Torch 1.13 safe) SDP fallback
            q, k, v = torch.split(qkv, 1, dim=2)
            q = q.squeeze(2)  # [B,S,H,D]
            k = k.squeeze(2)
            v = v.squeeze(2)

            # [B,H,S,D]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            attn_out = sdp_attention(q, k, v, attn_mask=None)  # [B,H,S,D]
            out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, S, -1)  # [B,S,C]

        out = self.out_proj(out)
        return out


# -----------------------------
# Transformer blocks
# -----------------------------
class NoTPTransformerBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, multiple_of=256):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        self.self_attn = NoTPAttention(cfg)
        self.mlp = NoTPFeedForward(cfg, dim=cfg.hidden_size, hidden_dim=cfg.ffn_hidden_size)

        self.layer_id = layer_id
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)

    def forward(self, x: torch.Tensor):
        residual = self.self_attn(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp(self.layer_norm2(h))
        return out


class NoTPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_layers = cfg.num_layers

        self.layers = nn.ModuleList([
            NoTPTransformerBlock(cfg, layer_id=i + 1) for i in range(self.num_layers)
        ])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# -----------------------------
# Top-level ViT Model
# -----------------------------
class LayerNormfp32(nn.LayerNorm):
    """LayerNorm in fp32 then cast back."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.to(torch.float32))
        return ret.to(orig_type)


class VitModel(nn.Module):
    def __init__(self, cfg, freeze_embed=False, freeze_pre_norm=False) -> None:
        super().__init__()

        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=cfg.hidden_size,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
        )

        if freeze_embed:
            for _, p in self.embeddings.named_parameters():
                p.requires_grad = False

        self.transformer = NoTPTransformer(cfg=cfg)

        if cfg.get("fp32norm", False):
            print("[clip_sdpa] Using fp32 LayerNorm for ViT.")
            self.pre_layrnorm = LayerNormfp32(cfg.hidden_size, eps=cfg.get("pre_layernorm_epsilon", 1e-5))
        else:
            self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.get("pre_layernorm_epsilon", 1e-5))

        if freeze_pre_norm:
            for _, p in self.pre_layrnorm.named_parameters():
                p.requires_grad = False

        # no-op flag from original codebase; harmless
        for p in self.parameters():
            p.micro_dp = True

    def __str__(self) -> str:
        return "open_clip"

    def forward(self, x: torch.Tensor, patch_embeds: Optional[torch.Tensor]):
        """
        x: [B,3,H,W]
        patch_embeds: [B, 1024, Hs, Ws] from SAM (preferred) or None
        returns: [B, 1+HW, 1024]
        """
        tokens = self.embeddings(x, patch_embeds)     # [B, 1+HW, 1024]
        hidden_states = self.pre_layrnorm(tokens)
        output = self.transformer(hidden_states)      # [B, 1+HW, 1024]
        return output


# -----------------------------
# Public builder
# -----------------------------
vit_model_cfg = adict(
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    seq_length=256,                 # not strictly used (only for legacy configs)
    max_position_embeddings=256,    # not strictly used
    use_flash_attn=False,           # set True only if flash-attn is installed
    understand_projector_stride=2,  # unused but kept for parity
    hidden_dropout=0.0,
    attention_dropout=0.0,
    no_persist_layer_norm=False,
    layernorm_epsilon=1e-5,
    pre_layernorm_epsilon=1e-5,
    image_size=224,
    patch_size=14,
    recompute_list=[],
)

def build_clip_l():
    return VitModel(
        cfg=vit_model_cfg,
        freeze_embed=False,
        freeze_pre_norm=False,
    )
