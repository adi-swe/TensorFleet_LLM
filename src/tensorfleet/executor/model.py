"""Small causal Transformer and stateless synthetic next-token batches."""

import torch
from torch import nn

from tensorfleet.config import TrainingConfig


class TinyTransformer(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.sequence_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            2 * config.d_model,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens):
        length = tokens.shape[1]
        hidden = self.embedding(tokens) + self.position(torch.arange(length, device=tokens.device))
        mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        return self.head(self.encoder(hidden, mask=mask))


def synthetic_batch(config: TrainingConfig, step: int, rank: int):
    # Independent streams indexed by global step/rank: no data cursor to lose on restart.
    generator = torch.Generator().manual_seed(config.seed + step * config.world_size + rank)
    start = torch.randint(config.vocab_size, (config.batch_size, 1), generator=generator)
    tokens = (start + torch.arange(config.sequence_length + 1)) % config.vocab_size
    return tokens[:, :-1], tokens[:, 1:]
