import math

import numpy as np
import torch
from torch import nn




class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim#更改处
        pos_encoding = torch.zeros(max_len, self.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * -(math.log(10000.0) / self.emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class MyEmbedding(nn.Module):
    def __init__(self, config):
        super(MyEmbedding, self).__init__()
        self.config = config

        self.num_locations = config.Dataset.num_locations
        self.num_locaiton_prototypes = 5
        self.base_dim = config.Embedding.base_dim
        self.num_users = config.Dataset.num_users

        
        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.prototypes_embedding = nn.Embedding(self.num_locaiton_prototypes, self.base_dim)
        self.loc_proj = nn.Linear(self.base_dim, self.base_dim)
        self.proto_proj = nn.Linear(self.base_dim, self.base_dim)
        self.softmax = nn.Softmax(dim=-1)


        self.timeslot_embedding = nn.Embedding(168, self.base_dim)




    def forward(self, batch_data):
        location_x = batch_data['location_x']


        location_all=self.location_embedding(torch.arange(end=self.num_locations, dtype=torch.int, device=location_x.device))

        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=168, dtype=torch.int, device=location_x.device))
        prototypes_embedded = self.prototypes_embedding(
                torch.arange(end=self.num_locaiton_prototypes, dtype=torch.int, device=location_x.device)
            )  # Shape: (num_prototypes, base_dim)
        loc_embedded = self.loc_proj(location_all)  # Shape: (num_locations, base_dim)
        proto_embedded = self.proto_proj(prototypes_embedded)  # Shape: (num_prototypes, base_dim)
        V = prototypes_embedded # Shape: (num_prototypes, base_dim)
        weight_scores = torch.matmul(loc_embedded, proto_embedded.T) / (self.base_dim ** 0.5)  # Scaled dot-product, Shape: (num_locations, num_prototypes)
        weights = self.softmax(weight_scores)  # Normalize scores, Shape: (num_locations, num_prototypes)
        loc_proto_repr = torch.matmul(weights, prototypes_embedded)  # Shape: (num_locations, base_dim)

        return location_all, timeslot_embedded, user_embedded, loc_proto_repr
