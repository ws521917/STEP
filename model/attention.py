import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.num_heads = 4
        self.head_dim = self.base_dim // self.num_heads
        self.num_locations = config.Dataset.num_locations
        self.num_locaiton_prototypes = 5

        self.locaiton_prototypes = nn.Embedding(self.num_locaiton_prototypes, self.base_dim)
        self.w_q = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_k = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_v = nn.ModuleList(
                [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

    def forward(self,combined_embedding,loc_proto_embedding,batch_data):
        user_x = batch_data['user']
        loc_x  = batch_data['location_x']
        batch_size, sequence_length = loc_x.shape

        K = loc_proto_embedding # (num_prototypes, embedding_dim)
        
        head_outputs = []
   
        query = combined_embedding


        for i in range(self.num_heads):
                query_i = self.w_q[i](query)
                key_i = self.w_k[i](K)
                value_i = self.w_v[i](K)
                attn_scores_i = torch.matmul(query_i, key_i.T)
                scale = 1.0 / (key_i.size(-1) ** 0.5)
                attn_scores_i = attn_scores_i * scale
                attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
                weighted_values_i = torch.matmul(attn_scores_i, value_i)
                head_outputs.append(weighted_values_i)
        head_outputs = torch.cat(head_outputs, dim=-1)
        head_outputs = head_outputs.view(batch_size, sequence_length, -1)
        return self.unify_heads(head_outputs)
