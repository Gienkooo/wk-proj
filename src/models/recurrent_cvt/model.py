from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import nn
from src.models.common.heads import MultiTaskHeads
from src.models.cvt.conv_tokenizer import ConvTokenizer
from src.models.cvt.block import TransformerBlock
from src.models.recurrent_cvt.cell import RecurrentCvTCell
from src.models.recurrent_cvt.unroll import unroll_cell

class RecurrentCvTModel(nn.Module):
    def __init__(self, in_channels, num_shapes, num_colors, num_materials,
                 embed_dim, depth, num_heads, mlp_ratio, dropout,
                 tokenizer_kernel, tokenizer_stride, recurrent_steps=4,
                 use_cls_token=True, **kwargs):
        super().__init__()

        self.recurrent_steps = recurrent_steps

        dim1 = max(16, embed_dim // 2)
        dim2 = dim3 = dim4 = embed_dim
        heads1 = max(1, num_heads // 2)
        heads2 = heads3 = heads4 = num_heads

        self.stage1_tok = ConvTokenizer(in_channels, dim1, tokenizer_kernel, tokenizer_stride)
        self.stage1_blk = TransformerBlock(dim1, heads1, mlp_ratio, dropout)

        self.stage2_tok = ConvTokenizer(dim1, dim2, kernel_size=3, stride=2)
        self.stage2_blk = TransformerBlock(dim2, heads2, mlp_ratio, dropout)

        self.stage3_tok = ConvTokenizer(dim2, dim3, kernel_size=3, stride=2)
        self.stage3_blk = TransformerBlock(dim3, heads3, mlp_ratio, dropout)

        self.stage4_tok = ConvTokenizer(dim3, dim4, kernel_size=3, stride=2)
        
        # Recurrent cell with exactly 1 shared block
        self.cell = RecurrentCvTCell(dim4, heads4, mlp_ratio, dropout, depth=1)

        self.pos4 = nn.Parameter(torch.zeros(1, dim4, 8, 8))
        nn.init.trunc_normal_(self.pos4, std=0.02)

        self.spatial_attn = nn.Conv2d(dim4, 1, kernel_size=1)
        self.spatial_proj = nn.Linear(dim4, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.heads = MultiTaskHeads(embed_dim, num_shapes=num_shapes,
                                    num_colors=num_colors, num_materials=num_materials,
                                    dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor] | None]:
        
        x = self.stage1_blk(self.stage1_tok(x))
        x = self.stage2_blk(self.stage2_tok(x))
        x = self.stage3_blk(self.stage3_tok(x))
        
        tokens = self.stage4_tok(x) + self.pos4
        
        embeddings = unroll_cell(self.cell, tokens, self.recurrent_steps)
        
        final_state = embeddings[-1]
        attn_map = torch.softmax(self.spatial_attn(final_state).flatten(2), dim=-1)
        weighted = (final_state.flatten(2) * attn_map).sum(dim=-1)
        latent = self.norm(self.spatial_proj(weighted))
        
        outputs = self.heads(latent)
        
        if return_embeddings:
            return outputs, embeddings
        return outputs, None