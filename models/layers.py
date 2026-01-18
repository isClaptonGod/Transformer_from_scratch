import torch.nn as nn
from models.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(query, key, value, mask)
        # Residual connection + Norm
        x = self.norm1(self.dropout(attention) + query)
        forward = self.feed_forward(x)
        out = self.norm2(self.dropout(forward) + x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, forward_expansion, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.transformer_block = TransformerBlock(d_model, num_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # Masked Multi-Head Attention
        attention = self.attention(x, x, x, trg_mask)
        query = self.norm(self.dropout(attention) + x)
        # Cross Attention with Encoder output
        out = self.transformer_block(value, key, query, src_mask)
        return out
