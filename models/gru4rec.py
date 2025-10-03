import torch
import torch.nn as nn
import numpy as np
from loguru import logger

class GRU4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dev = args.device
        self.hidden_units = args.hidden_units

        self.item_emb = nn.Embedding(args.itemnum1 + 1, self.hidden_units)
        self.num_layers = args.num_layers

        self.gru_layers = nn.GRU(
            input_size=self.hidden_units,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(args.dropout_rate)
        self.dense = nn.Linear(self.hidden_units, self.hidden_units)

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()   

    def gru4rec(self, item_seq):
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def forward(self, user_ids, log_seqs, log_seq2, mask ):
        
        logits = self.gru4rec(torch.LongTensor(log_seqs).to(self.dev))
        
        return logits

    def predict(self, targetSeq, test_item):
        logits = self.gru4rec(torch.LongTensor(targetSeq).to(self.dev))

        # embedding_weight = self.item_emb.weight
        test_item_in = self.item_emb(torch.LongTensor(test_item).to(self.dev))

        final_feat = logits[:, -1, :]  # 1 x h
        test_logits = test_item_in.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return test_logits

    ## require
    def calculate_loss(self, batch):
        _, seq, pos, neg, _, _,_,_ = batch
        seq_out = self.gru4rec(torch.LongTensor(seq).to(self.dev))
        
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
        
        seq_out = self.gru4rec(torch.LongTensor(np.array(seq)).to(self.dev))
        seq_out = seq_out[:, -1, :]
        
        # [item_num hidden_size]
        test_item_emb = self.item_emb.weight[1:self.args.itemnum1+1,:]
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    