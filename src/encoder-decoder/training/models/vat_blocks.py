"""VAT Transformer Block :: vat_blocks.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Check for Flash Attention availability
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

# Check for PyTorch SDPA availability (PyTorch 2.0+)
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")

# One-time warnings to avoid log spam
_WARNED_FLASH_DTYPE = False
_WARNED_FLASH_CPU = False
_WARNED_NO_SDPA = False


def _chunked_attention(q, k, v, dropout_p=0.0):
    """
    Manual chunked attention fallback for PyTorch < 2.0.
    q, k, v: [B, H, S, D]
    Returns: [B, H, S, D]
    """
    global _WARNED_NO_SDPA
    if not _WARNED_NO_SDPA:
        print("[vat_blocks] ⚠️  Using chunked attention fallback (PyTorch < 2.0). "
              "Consider upgrading to PyTorch 2.0+ for faster SDPA.")
        _WARNED_NO_SDPA = True
    
    B, H, S, D = q.shape
    chunk_size = min(1024, S)
    dk = D ** 0.5
    outputs = []
    
    for i in range(0, S, chunk_size):
        q_chunk = q[:, :, i:i+chunk_size, :]  # [B, H, chunk, D]
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / dk  # [B, H, chunk, S]
        attn = torch.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        out_chunk = torch.matmul(attn, v)  # [B, H, chunk, D]
        outputs.append(out_chunk)
    
    return torch.cat(outputs, dim=2)  # [B, H, S, D]


class FlashMultiheadAttention(nn.Module):
    """
    Multi-head attention with optional Flash Attention support.
    Falls back to PyTorch's scaled_dot_product_attention when Flash Attention is not available.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, is_cross_attn: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.is_cross_attn = is_cross_attn
        
        if is_cross_attn:
            # Separate projections for cross-attention
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        else:
            # Combined QKV projection for self-attention
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: Query tensor [B, Sq, D]
            k: Key tensor [B, Sk, D] (for cross-attention) or None (for self-attention)
            v: Value tensor [B, Sk, D] (for cross-attention) or None (for self-attention)
        Returns:
            Output tensor [B, Sq, D]
        """
        B, Sq, _ = q.shape
        
        if self.is_cross_attn:
            # Cross-attention
            assert k is not None and v is not None
            Sk = k.shape[1]
            q_proj = self.q_proj(q).view(B, Sq, self.n_heads, self.head_dim)
            k_proj = self.k_proj(k).view(B, Sk, self.n_heads, self.head_dim)
            v_proj = self.v_proj(v).view(B, Sk, self.n_heads, self.head_dim)
        else:
            # Self-attention
            qkv = self.qkv_proj(q).view(B, Sq, 3, self.n_heads, self.head_dim)
            q_proj = qkv[:, :, 0]  # [B, Sq, H, D]
            k_proj = qkv[:, :, 1]
            v_proj = qkv[:, :, 2]
        
        # Determine attention implementation to use
        # Priority: Flash Attention > PyTorch SDPA > Chunked fallback
        global _WARNED_FLASH_DTYPE, _WARNED_FLASH_CPU
        
        use_flash = False
        if _HAS_FLASH_ATTN:
            if not q.is_cuda:
                if not _WARNED_FLASH_CPU:
                    print("[vat_blocks] ⚠️  Flash Attention available but inputs on CPU. Using SDPA fallback.")
                    _WARNED_FLASH_CPU = True
            elif q.dtype not in (torch.float16, torch.bfloat16):
                # Auto-cast float32 inputs to bfloat16 for Flash Attention compatibility
                # This happens when autocast context hasn't propagated to input tensors yet
                # (e.g., inputs loaded from numpy as float32)
                target_dtype = torch.bfloat16  # bf16 is more stable than fp16
                q_proj = q_proj.to(target_dtype)
                k_proj = k_proj.to(target_dtype)
                v_proj = v_proj.to(target_dtype)
                use_flash = True
                if not _WARNED_FLASH_DTYPE:
                    print(f"[vat_blocks] ℹ️  Auto-casting {q.dtype} inputs to {target_dtype} for Flash Attention.")
                    _WARNED_FLASH_DTYPE = True
            else:
                use_flash = True
        
        dropout_p = self.dropout if self.training else 0.0
        
        if use_flash:
            # Flash Attention expects [B, S, H, D] format
            out = flash_attn_func(q_proj, k_proj, v_proj, dropout_p=dropout_p, causal=False)
            out = out.view(B, Sq, self.d_model)
        else:
            # Transpose to [B, H, S, D] for SDPA/chunked attention
            q_proj = q_proj.transpose(1, 2)
            k_proj = k_proj.transpose(1, 2)
            v_proj = v_proj.transpose(1, 2)
            
            if _HAS_SDPA:
                # Use PyTorch's memory-efficient SDPA (PyTorch 2.0+)
                out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=dropout_p)
            else:
                # Fallback: chunked manual attention for PyTorch < 2.0
                out = _chunked_attention(q_proj, k_proj, v_proj, dropout_p=dropout_p)
            
            out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        
        return self.out_proj(out)


class VATBlock(nn.Module):
    """
    Transformer block with SA + Cross-Attn (query attends to kv) + MLP
    Now with optional Flash Attention support for faster training.
    Supports gradient checkpointing for memory efficiency.
    
    Shapes:
      q:  [B, nq, d_model]
      kv: [B, N_kv, d_model]
      out:[B, nq, d_model]
    """
    
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float):
        super().__init__()
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa = FlashMultiheadAttention(d_model, n_heads, dropout=dropout, is_cross_attn=False)
        
        self.ca_ln = nn.LayerNorm(d_model)
        self.ca = FlashMultiheadAttention(d_model, n_heads, dropout=dropout, is_cross_attn=True)
        
        self.mlp_ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def _sa_forward(self, q: torch.Tensor) -> torch.Tensor:
        """Self-attention sub-block (for gradient checkpointing)"""
        q_norm = self.sa_ln(q)
        return self.sa(q_norm)
    
    def _ca_forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Cross-attention sub-block (for gradient checkpointing)"""
        q_norm = self.ca_ln(q)
        return self.ca(q_norm, kv, kv)
    
    def _mlp_forward(self, q: torch.Tensor) -> torch.Tensor:
        """MLP sub-block (for gradient checkpointing)"""
        return self.mlp(self.mlp_ln(q))
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            # use_reentrant=False is more memory efficient and handles autocast properly
            q = q + checkpoint(self._sa_forward, q, use_reentrant=False)
            q = q + checkpoint(self._ca_forward, q, kv, use_reentrant=False)
            q = q + checkpoint(self._mlp_forward, q, use_reentrant=False)
        else:
            # Standard forward pass
            q_norm = self.sa_ln(q)
            q = q + self.sa(q_norm)
            
            q_norm = self.ca_ln(q)
            q = q + self.ca(q_norm, kv, kv)
            
            q = q + self.mlp(self.mlp_ln(q))
        
        return q
