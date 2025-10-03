# @Time   : 2022/2/13
# @Author : Hui Yu
# @Email  : ishyu@outlook.com

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
import numpy as np
from loguru import logger

class FMLP(nn.Module):
    def __init__(self, args):
        super(FMLP, self).__init__()
        self.args = args
        self.item_num = args.itemnum1
        self.item_emb = nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.maxlen, args.hidden_units)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.item_encoder = Encoder(args)

        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_emb(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        input_ids = torch.LongTensor(input_ids).to(self.args.device)
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(self.args.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


        
    ## require
    def calculate_loss(self, batch):
        _, seq, pos, neg, _, _,_,_ = batch
        seq_out = self.forward(seq)
        
        # only use the cross_entropy
        loss = self.cross_entropy(seq_out,pos,neg)
    
        return loss
   

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_ids = torch.LongTensor(pos_ids).to(self.args.device)
        neg_ids = torch.LongTensor(neg_ids).to(self.args.device)
        # [batch seq_len hidden_size]
        pos_emb = self.item_emb(pos_ids)
        neg_emb = self.item_emb(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.maxlen).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss


    ## require
    def predict_full(self, batch):
        u, seq, seq2, mask, items,_,_,_ = batch
        
        seq_out = self.forward(np.array(seq))
        seq_out = seq_out[:, -1, :]
        
        # [item_num hidden_size]
        test_item_emb = self.item_emb.weight[1:self.item_num+1,:]
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred 
