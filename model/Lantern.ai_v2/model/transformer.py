import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================== RoPE ==========================

# (Unchanged, included for completeness)
def get_rotary_matrix(seq_len, d_model, device, theta_base=10000.0, eps=1e-6):
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    theta = 1.0 / (theta_base ** (2.0 * torch.arange(0, d_model, 2, device=device).float() / d_model + eps))
    positions = torch.arange(seq_len, device=device).float().unsqueeze(1)
    angles = positions * theta
    angles = torch.clamp(angles, min=-1e4, max=1e4)

    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    rotary_matrix = torch.stack([cosines, -sines, sines, cosines], dim=-1)
    rotary_matrix = rotary_matrix.view(seq_len, d_model // 2, 2, 2)
    return rotary_matrix

def apply_rotary_embeddings(x, rotary_matrix):
    batch_size, seq_len, d_model = x.shape
    rot_seq_len, rot_dim, _, _ = rotary_matrix.shape

    if rot_seq_len < seq_len:
        raise ValueError(f"Rotary matrix seq_len {rot_seq_len} is too short for input seq_len {seq_len}")
    if rot_dim != d_model // 2:
        raise ValueError(f"Rotary matrix dim {rot_dim} does not match d_model // 2 = {d_model // 2}")

    x_reshaped = x.view(batch_size, seq_len, d_model // 2, 2)
    rotary_matrix = rotary_matrix[:seq_len].unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    x_rotated = torch.einsum('bsdh,bsdij->bsdh', x_reshaped, rotary_matrix)
    return x_rotated.view(batch_size, seq_len, d_model)

# ========================== TRANSFORMER BLOCK ==========================

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(dtype=torch.bool)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=torch.bool)
        attn_output, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )

        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # Feedforward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ff_output))

        return tgt

class StoryTellerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for RoPE, got {d_model}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        batch_size, seq_len = tgt.shape
        device = tgt.device

        x = self.embedding(tgt)
        rotary_matrix = get_rotary_matrix(seq_len, self.embedding.embedding_dim, device)
        x = apply_rotary_embeddings(x, rotary_matrix)

        for layer in self.decoder_layers:
            x = layer(tgt=x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
