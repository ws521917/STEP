import math

import torch

from torch import nn

device = torch.device("cuda")

from embedding import MyEmbedding, PositionalEncoding
from encoder import TransEncoder
from fullyconnect import MyFullyConnect,MyFullyConnect_Fusion


from attention import Attention



class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.num_locations = config.Dataset.num_locations
        self.num_users =config.Dataset.num_users
        self.embedding_layer = MyEmbedding(config)

        if config.Encoder.encoder_type == 'trans':
            emb_dim = self.base_dim
            self.positional_encoding = PositionalEncoding(emb_dim=emb_dim)
            self.positional_encoding_time = PositionalEncoding(emb_dim=emb_dim)
            self.encoder = TransEncoder(config,emb_dim)
            self.encoder_time = TransEncoder(config,emb_dim)

        fc_input_dim = self.base_dim + self.base_dim   #  --------------------维度更改在这里----------------------------
        fc_input_dim_time = self.base_dim + self.base_dim 


        self.attention=Attention(config=config)

        self.linear=nn.Linear(fc_input_dim,self.base_dim)
        fc_input_dim =fc_input_dim + self.base_dim


        self.fc_layer = MyFullyConnect(input_dim=fc_input_dim, output_dim=2048)

        self.fc_layer_time = MyFullyConnect(input_dim=fc_input_dim_time,output_dim=168)

        self.fc_layer_all = MyFullyConnect_Fusion(input_dim=2048+168,output_dim=config.Dataset.num_locations)



    def forward(self, batch_data):


        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        time_y = batch_data['timeslot_y']
        batch_size, sequence_length = loc_x.shape




        
        loc_embedded_yuan, timeslot_embedded,user_embedded_yuan, locaiton_category_embedded = self.embedding_layer(batch_data)

        time_embedded = timeslot_embedded[hour_x]
        loc_embedded = loc_embedded_yuan[loc_x]
        location_category_embedding = locaiton_category_embedded[loc_x]
        



        lt_embedded = loc_embedded + time_embedded 

        
        if self.config.Encoder.encoder_type == 'trans':
            future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(lt_embedded.device)
            future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
            encoder_out = self.encoder(self.positional_encoding(lt_embedded * math.sqrt(self.base_dim)),
                                       src_mask=future_mask)

        combined = encoder_out


        user_embedded = user_embedded_yuan[user_x]


        user_embedded = user_embedded.unsqueeze(1).repeat(1, sequence_length, 1)
        combined=torch.cat([combined, user_embedded], dim=-1)

# ---------------------------------------user embedding------------------------------------------------------------------

        user_embedded_exp = user_embedded + time_embedded 


        future_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(user_embedded.device)
        future_mask = future_mask.masked_fill(future_mask == 1, float('-inf')).bool()
        encoder_out_time = self.encoder_time(self.positional_encoding_time(user_embedded_exp * math.sqrt(self.base_dim)),
                                             src_mask=future_mask)
        user_time_embedded = encoder_out_time



        user_time_embedded=torch.cat([user_time_embedded, loc_embedded+location_category_embedding], dim=-1)

#-----------------------------------------------------------------------------------------------------------------------


        combined_cf=self.linear(combined)

        attention_output=self.attention(combined_cf,locaiton_category_embedded,batch_data)

        combined=torch.cat([combined, attention_output], dim=-1)


        out_loc = self.fc_layer(combined.view(batch_size * sequence_length, -1))

        out_time =self.fc_layer_time(user_time_embedded.view(batch_size * sequence_length, -1))

        last_input=torch.cat([out_loc, out_time], dim=-1)
        last_output =self.fc_layer_all(last_input)

        return last_output,out_time
