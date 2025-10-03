import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.layer import MultiHeadAttention
from loguru import logger


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class multihead_attention(torch.nn.Module):
    def __init__(self,  args):
        super(multihead_attention, self).__init__()

        self.dev = args.device
        self.num_blocks = args.num_blocks
        self.first_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.query_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.key_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.key_attention_layernorms = torch.nn.ModuleList()
        self.value_attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.layer_norm_eps = 1e-8

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            key_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.key_attention_layernorms.append(key_attn_layernorm)

            value_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.value_attention_layernorms.append(value_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.num_heads, args.hidden_units, args.dropout_rate,
                                                args.dropout_rate, self.layer_norm_eps, args.maxlen)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def get_attn_mask(self, attention_mask, queries):
        new_attn_mask = torch.zeros_like(attention_mask, dtype=queries.dtype)
        new_attn_mask.masked_fill_(attention_mask, float("-inf"))
        return new_attn_mask

    def seq_log2feats(self, queries, timeline_mask):
        queries = self.emb_dropout(queries)
        queries *= ~timeline_mask.unsqueeze(-1)

        tl = queries.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        attention_mask = self.get_attn_mask(attention_mask, queries)

        for i in range(self.num_blocks):
            Q = self.attention_layernorms[i](queries)
            queries, attention_map = self.attention_layers[i](Q, queries, queries, attention_mask)
            queries += Q
            queries = self.forward_layernorms[i](queries)
            queries = self.forward_layers[i](queries)
            queries *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(queries)
        return log_feats


    def forward(self, queries, timeline_mask):
        return self.seq_log2feats(queries, timeline_mask)


class CDSASRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dev = args.device

        self.hidden_units = args.hidden_units
        self.item_emb = nn.Embedding(args.itemnum + 1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.multihead_attention_layers = multihead_attention(args)
        self.multihead_attention_layers_source = multihead_attention(args)
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        if self.args.init_type == "normal":
            # logger.info("use the normal init ")
            # 原始的初始化方法
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        else:
            # Xavier初始化方法
            logger.info("use the xavier init ")
            for name, param in self.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass

    def seq_self_attention_layer(self, targetSeq, sourceSeq):

        has_sourceSeq = not np.all(sourceSeq == 0)
        # 如果没有源领域，那么就跳过这部分。
        if  has_sourceSeq:
            source_timeline_mask = torch.BoolTensor(sourceSeq == 0).to(self.dev)

            sourceEmb = self.item_emb(torch.LongTensor(sourceSeq).to(self.dev))  # 1 x item_num x h

            positions = np.tile(np.array(range(sourceEmb.shape[1])), [sourceEmb.shape[0], 1])
            sourceSeqs = sourceEmb * (self.item_emb.embedding_dim ** 0.5)
            sourceSeqs = sourceSeqs + self.pos_emb(torch.LongTensor(positions).to(self.dev))
            sourceSeqs = self.multihead_attention_layers_source(sourceSeqs, source_timeline_mask)

            sourceSeqs = sourceSeqs[:,-1,:]
            sourceSeqs = sourceSeqs.unsqueeze(1)
            sourceSeqs = sourceSeqs.repeat(1, targetSeq.shape[1], 1)

        target_timeline_mask = torch.BoolTensor(targetSeq == 0).to(self.dev)

        targetEmb = self.item_emb(torch.LongTensor(targetSeq).to(self.dev))  # 1 x item_num x h

        positions = np.tile(np.array(range(targetEmb.shape[1])), [targetEmb.shape[0], 1])
        targetSeqs = targetEmb * (self.item_emb.embedding_dim ** 0.5)
        if has_sourceSeq:
            # follow the paper equation to do the early fusion
            targetSeqs = sourceSeqs + targetSeqs + self.pos_emb(torch.LongTensor(positions).to(self.dev))

            # fix up the early fusion part : target * source + position from the paper main architecture diagram
            # targetSeqs = targetSeqs * sourceSeqs + self.pos_emb(torch.LongTensor(positions).to(self.dev))
            
            # targetSeqs = targetSeqs + (self.pos_emb(torch.LongTensor(positions).to(self.dev)) * sourceSeqs)
            
        targetSeqs = self.multihead_attention_layers(targetSeqs, target_timeline_mask)

        return targetSeqs

    def forward(self,  user_ids, log_seqs1, log_seqs2, mask):

        logits = self.seq_self_attention_layer(log_seqs1, log_seqs2)

        return logits

    def predict(self, targetSeq, sourceSeq, test_item):
        seq_sasrec = self.seq_self_attention_layer(targetSeq, sourceSeq)

        # embedding_weight = self.item_emb.weight
        test_item_in = self.item_emb(torch.LongTensor(test_item).to(self.dev))

        final_feat = seq_sasrec[:, -1, :]  # 1 x h
        test_logits = test_item_in.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return test_logits

    
    ## require
    def calculate_loss(self, batch):
        u, seq, pos, neg, seq2, _ ,mixed_seq,mixed_time,_= batch
        # seq_out = self.log2feats(seq)
        seq_out = self.seq_self_attention_layer(seq, seq2)
        
        if self.args.loss_type == "CE":
            pos = torch.LongTensor(pos).to(self.dev)
            test_item_emb = self.item_emb.weight[1:self.args.itemnum1 + 1, :] # 物品编码是从1开始的，0是padding值。
            # 这里需要保留padding进行计算，不然会导致CEloss找不到对应位置。即映射概率值是从0开始，但是pos的标签是从1开始索引的。
            logits = seq_out @ test_item_emb.T
            indices = torch.where(pos.view(-1) != 0)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos.view(-1)[indices]-1, reduction='mean')
            
        else:
            loss = self.cross_entropy(seq_out,pos,neg)
    
        return loss
   

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_ids = torch.LongTensor(pos_ids).to(self.dev)
        neg_ids = torch.LongTensor(neg_ids).to(self.dev)
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


    def predict_full(self, batch):
        u, seq, seq2, mask, item_idx, items,mixed_seq,mixed_time = batch
        seq_out = self.seq_self_attention_layer(np.array(seq), np.array(seq2))[:, -1, :]

        if self.args.sample_eval:
            test_item_emb = self.item_emb(torch.LongTensor(np.array(item_idx)).to(self.dev))

            rating_pred = torch.matmul(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)

        else:
            test_item_emb = self.item_emb.weight[1:self.args.itemnum1+1,:]
            rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))

        return rating_pred
