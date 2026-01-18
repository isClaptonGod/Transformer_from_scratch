import torch
import torch.nn as nn
from models.layers import TransformerBlock, DecoderBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
                 d_model=512, num_layers=6, forward_expansion=4, num_heads=8, dropout=0.1, max_len=100):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, forward_expansion, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len).to(trg.device)
        return trg_mask & (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.dropout(self.pos_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, enc_src, enc_src, src_mask)
            
        out = self.dropout(self.pos_encoding(self.decoder_embedding(trg)))
        for layer in self.decoder_layers:
            out = layer(out, enc_src, enc_src, src_mask, trg_mask)
            
        return self.fc_out(out)
