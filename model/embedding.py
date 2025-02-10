import math

import numpy as np
import torch
from torch import nn



from time_pre import GATModel


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
        self.num_locaiton_category = 5
        self.num_time_category = 4
        self.base_dim = config.Embedding.base_dim
        self.num_users = config.Dataset.num_users

        
        self.user_embedding = nn.Embedding(self.num_users, self.base_dim)
        self.location_embedding = nn.Embedding(self.num_locations, self.base_dim)
        self.cat_embedding = nn.Embedding(self.num_locaiton_category, self.base_dim)
        self.query_proj = nn.Linear(self.base_dim, self.base_dim)
        self.key_proj = nn.Linear(self.base_dim, self.base_dim)
        self.value_proj = nn.Linear(self.base_dim, self.base_dim)
        self.softmax = nn.Softmax(dim=-1)


        self.timeslot_embedding = nn.Embedding(168, self.base_dim)




    def forward(self, batch_data):
        location_x = batch_data['location_x']


        location_all=self.location_embedding(torch.arange(end=self.num_locations, dtype=torch.int, device=location_x.device))

        user_embedded = self.user_embedding(torch.arange(end=self.num_users, dtype=torch.int, device=location_x.device))

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=168, dtype=torch.int, device=location_x.device))
        cat_embedded = self.cat_embedding(
                torch.arange(end=self.num_locaiton_category, dtype=torch.int, device=location_x.device)
            )  # Shape: (num_cats, base_dim)
        Q = self.query_proj(location_all)  # Shape: (num_locations, base_dim)
        K = self.key_proj(cat_embedded)  # Shape: (num_cats, base_dim)
        V = cat_embedded # Shape: (num_cats, base_dim)
        attn_scores = torch.matmul(Q, K.T) / (self.base_dim ** 0.5)  # Scaled dot-product, Shape: (num_locations, num_cats)
        attn_weights = self.softmax(attn_scores)  # Normalize scores, Shape: (num_locations, num_cats)
        loc_cat_repr = torch.matmul(attn_weights, V)  # Shape: (num_locations, base_dim)

        return location_all, timeslot_embedded, user_embedded, loc_cat_repr
