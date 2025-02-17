import torch.nn as nn
from torch.nn import init


class TransEncoder(nn.Module):
    def __init__(self, config,embedding_dim):
        super(TransEncoder, self).__init__()
        self.config = config
        input_dim = embedding_dim#更改

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dim_feedforward=input_dim,
                                                   nhead=4,
                                                   dropout=0.1)

        encoder_norm = nn.LayerNorm(input_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=2,
                                             norm=encoder_norm)
        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        out = self.encoder(embedded_out, mask=src_mask)

        return out

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)



