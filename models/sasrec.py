import torch
from modules import PointWiseFeedForward
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from loguru import logger
import copy
import random
class SASRec(torch.nn.Module):
    def __init__(self,args):
        super(SASRec, self).__init__()

        self.user_num = args.usernum1
        self.item_num = args.itemnum1
        self.dev = args.device
        self.args = args
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
       

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
            
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
 
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                           args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if self.args.init_type == "normal":
            # logger.info("use the normal init ")
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        else:
            # logger.info("use the xavier init ")
            for name, param in self.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass
            
                 
    def log2feats(self, log_seqs):


        # log_seqs=self.shuffle_items(log_seqs) # random

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        seqs = seqs *( self.item_emb.embedding_dim ** 0.5)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):

            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)

            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)

            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)


        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        # print(log_feats.shape)
        return log_feats

    def forward(self, user_ids, log_seqs, log_seq2, mask): # for training    

        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        return log_feats
    
    ## require
    def calculate_loss(self, batch):
        _, seq, pos, neg, _, _,_,_,_ = batch
        seq_out = self.log2feats(seq)
        
        if self.args.loss_type == "CE":
            pos = torch.LongTensor(pos).to(self.args.device)
            test_item_emb = self.item_emb.weight[1:self.item_num + 1, :]
            logits = seq_out @ test_item_emb.T
            indices = torch.where(pos.view(-1) != 0)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], (pos.view(-1)[indices]-1), reduction='mean')
            
            return loss
        # only use the cross_entropy
        else:
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
        # u, seq, seq2, mask, sampled_items, items, mark_items = batch
        seq_out = self.log2feats(np.array(batch[1]))[:, -1, :]

        if self.args.sample_eval:
            item_idx = batch[4]
            try:
                test_item_emb = self.item_emb(torch.LongTensor(np.array(item_idx)).to(self.dev))
                # test_item_emb = self.item_emb.weight[np.array(item_idx)]
            except Exception as e:
                print(f"Error with item_idx: {item_idx}")
                print(f"item_idx shape: {np.array(item_idx).shape}, dtype: {np.array(item_idx).dtype}")
                raise e
            rating_pred = torch.matmul(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        else:
            test_item_emb = self.item_emb.weight[1:self.args.itemnum1+1,:]
            rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def shuffle_items(self,Seq):
        new_Seq=copy.deepcopy(Seq)
        for i in range(Seq.shape[0]):
            new_Seq[i]=random.sample(list(new_Seq[i]), len(new_Seq[i]))

        return new_Seq
