import torch
from torch import nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # padding=1 is critical here as the zeros act as spatial border anchors
        self.q_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=num_heads)
        self.k_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=num_heads)
        self.v_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=num_heads)

        self.q_bn = nn.BatchNorm2d(embed_dim)
        self.k_bn = nn.BatchNorm2d(embed_dim)
        self.v_bn = nn.BatchNorm2d(embed_dim)

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. Spatial Convolution for Q, K, V
        q = self.q_bn(self.q_conv(x)).flatten(2).transpose(1, 2)
        k = self.k_bn(self.k_conv(x)).flatten(2).transpose(1, 2)
        v = self.v_bn(self.v_conv(x)).flatten(2).transpose(1, 2)
        
        # 2. Reshape for multi-head attention
        q = q.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        # 4. Reconstruct spatial 2D feature map
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(out)
