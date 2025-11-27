"""VAT Transformer Block :: vat_blocks.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for Flash Attention availability
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


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
        
        # Use Flash Attention if available and input is on CUDA with compatible dtype
        use_flash = (
            _HAS_FLASH_ATTN and 
            q.is_cuda and 
            q.dtype in (torch.float16, torch.bfloat16)
        )
        
        if use_flash:
            # Flash Attention expects [B, S, H, D] format
            dropout_p = self.dropout if self.training else 0.0
            out = flash_attn_func(q_proj, k_proj, v_proj, dropout_p=dropout_p, causal=False)
            out = out.view(B, Sq, self.d_model)
        else:
            # Fall back to PyTorch SDPA
            # Transpose to [B, H, S, D] for SDPA
            q_proj = q_proj.transpose(1, 2)
            k_proj = k_proj.transpose(1, 2)
            v_proj = v_proj.transpose(1, 2)
            
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, dropout_p=dropout_p)
            out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        
        return self.out_proj(out)


class VATBlock(nn.Module):
    """
    Transformer block with SA + Cross-Attn (query attends to kv) + MLP
    Now with optional Flash Attention support for faster training.
    
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
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q_norm = self.sa_ln(q)
        q = q + self.sa(q_norm)
        
        # Cross-attention
        q_norm = self.ca_ln(q)
        q = q + self.ca(q_norm, kv, kv)
        
        # MLP
        q = q + self.mlp(self.mlp_ln(q))
        
        return q
