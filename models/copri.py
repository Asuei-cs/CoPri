
from models.layer import MultiHeadAttention
from args import *
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,dropout_rate=0.5):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


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
    def __init__(self, args):
        super(multihead_attention, self).__init__()

        self.dev = args.device
        self.num_blocks = args.t_num_blocks
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
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


    def forward(self, queries, timeline_mask):
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

class CrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.dev=args.device

        self.num_blocks = args.s_num_blocks

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(args.hidden_units, args.num_heads, dropout=args.dropout_rate, batch_first=True)
            for _ in range(self.num_blocks)
        ])
        self.attn_layernorms = nn.ModuleList([
            nn.LayerNorm(args.hidden_units, eps=1e-8)
            for _ in range(self.num_blocks)
        ])
        self.ffn_layers = nn.ModuleList([
            PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            for _ in range(self.num_blocks)
        ])
        self.forward_layernorms = nn.ModuleList([
            nn.LayerNorm(args.hidden_units, eps=1e-8)
            for _ in range(self.num_blocks)
        ])
        self.dropout = nn.Dropout(args.dropout_rate)


    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        x = query

        for i in range(self.num_blocks):
            attn_output, _ = self.attn_layers[i](x, key, value, key_padding_mask=key_padding_mask,attn_mask=attn_mask)
            x = self.attn_layernorms[i](x + self.dropout(attn_output))
            ffn_output = self.dropout(self.ffn_layers[i](x))
            x = self.forward_layernorms[i](x + ffn_output)
        return x

class AUG(torch.nn.Module):
    def __init__(self, args):
        super(AUG, self).__init__()
        self.device = args.device
        self.G= nn.Parameter(torch.randn(args.batch_size, args.maxlen,args.hidden_units))
        nn.init.kaiming_normal_(self.G)
    def generator(self,Seqs,size_0):
        Seqs = Seqs.to(self.device)
        G_data= self.G * Seqs.mean(dim=0, keepdim=True)
        G_data = G_data[:size_0, :, :]
        self.new = G_data

        return self.new






class CoPri(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dev = args.device
        self.itemnum1=args.itemnum1
        self.maxlen=args.maxlen

        self.hidden_units = args.hidden_units
        self.item_emb = nn.Embedding(args.itemnum + 1, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(2*args.maxlen, args.hidden_units)
        self.multihead_attention_layers = multihead_attention(args).to(self.dev)
        # self.multihead_attention_layers_source = multihead_attention(args).to(self.dev)
        self.multihead_attention_layers_sub=multihead_attention(args).to(self.dev)
        self.apply(self._init_weights)

        self.mlp2 = MLP(args.hidden_units*2, args.hidden_units,args.dropout_rate).to(self.dev)
        self.aug=AUG(args).to(self.dev)
        self.cross_attention=CrossAttention(args).to(self.dev)

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
            for name, param in self.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass


    def decompose_sequence(self, mixSeqs):
        source_sub_seqs = []
        source_sub_positions = []
        target_item_pos = []
        target_item_positions = []
        for sequence in mixSeqs:
            sequence = np.array(sequence)
            target_indices = np.where((sequence >= 1) & (sequence <= self.itemnum1))[0]
            for idx in target_indices:
                source_items = sequence[:idx][sequence[:idx] > self.itemnum1]
                if len(source_items) == 0:
                    continue
                if len(source_items) > self.maxlen:
                    sub_seq = source_items[-self.maxlen:]
                else:
                    pad_len = self.maxlen - len(source_items)
                    sub_seq = np.pad(source_items, (pad_len, 0), 'constant', constant_values=0)
                source_sub_seqs.append(sub_seq.tolist())
                t_item = sequence[idx]
                t_pos = idx
                target_item_pos.append([t_item])
        target_item_pos = [np.pad(x, (self.maxlen - len(x), 0), 'constant', constant_values=0) for x in target_item_pos]

        target_item_pos = np.array(target_item_pos)
        return (
            torch.tensor(source_sub_seqs, dtype=torch.long),
            torch.tensor(source_sub_positions, dtype=torch.long),
            torch.tensor(target_item_pos, dtype=torch.long),
            torch.tensor(target_item_positions, dtype=torch.long)
        )


    def get_embeddings(self,Seq,need_position=True,position=None):

        Emb = self.item_emb(torch.LongTensor(Seq).to(self.dev))  # 1 x item_num x h
        Seq_e = Emb * (self.item_emb.embedding_dim ** 0.5)
        if need_position:
            if position is not None:
                Seq_e = Seq_e + self.pos_emb(torch.LongTensor(position).to(self.dev))
            else:
                positions = np.tile(np.array(range(Emb.shape[1])), [Emb.shape[0], 1])
                Seq_e = Seq_e + self.pos_emb(torch.LongTensor(positions).to(self.dev))

        return Seq_e


    def seq_self_attention_layer(self, user,targetSeq, sourceSeq,mixSeq):

        target_timeline_mask = torch.BoolTensor(targetSeq == 0).to(self.dev)
        target_e=self.get_embeddings(targetSeq)
        target_h = self.multihead_attention_layers(target_e, target_timeline_mask)


        # source_timeline_mask = torch.BoolTensor(sourceSeq == 0).to(self.dev)
        source_e=self.get_embeddings(sourceSeq)


        source_sub_seqs,source_position,target_item_pos,target_position=self.decompose_sequence(mixSeq)
        target_item_pos=target_item_pos[:,-1].unsqueeze(-1)
        subseqs_e=self.get_embeddings(source_sub_seqs)
        target_item_pos_e=self.get_embeddings(target_item_pos, need_position=False)
        source_h_cross=self.cross_attention(subseqs_e,target_item_pos_e,target_item_pos_e)

        source_h_g = self.aug.generator(source_h_cross,source_e.size(0))
        source_h=source_h_g
        g_loss = torch.log1p(self.huber_loss(source_h_g, target_h))


        mixed_out = torch.cat((target_h, source_h), dim=-1)  # batch x maxlen x 2h
        mixed_out = self.mlp2(mixed_out)#batch x maxlen x h
        mixed_out=F.dropout(mixed_out, p=self.args.dropout_rate, training=self.training)

        loss= g_loss

        return target_h,mixed_out,source_h,loss





    def calculate_loss(self, batch):
        u, seq, pos, neg, seq2, _ ,mixed_seq,mixed_time,_= batch

        target_out,mixed_out,source_out,loss_n= self.seq_self_attention_layer(u,seq, seq2,mixed_seq)
        seq_out=mixed_out

        if self.args.loss_type == "CE":
            loss_1=torch.log1p(self.CE_loss(mixed_out, pos))
            loss_2=torch.log1p(self.CE_loss(target_out, pos))
            loss=self.args.w1*loss_1+self.args.w2*loss_2+self.args.w3*loss_n


        else:
            loss = self.BCE_loss(seq_out, pos, neg)
        return loss

    def CE_loss(self,seq_out,pos):
        pos = torch.LongTensor(pos).to(self.args.device)
        test_item_emb = self.item_emb.weight[1:self.args.itemnum1 + 1, :]
        logits = seq_out @ test_item_emb.T
        indices = torch.where(pos.view(-1) != 0)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos.view(-1)[indices] - 1,
                               reduction='mean')
        return loss

    def huber_loss(self, seqs1,seqs2):

        loss = F.smooth_l1_loss(seqs1, seqs2, reduction='mean')

        return  loss


    def BCE_loss(self, seq_out, pos_ids, neg_ids):
        pos_ids = torch.as_tensor(pos_ids, dtype=torch.long, device=self.args.device)
        neg_ids = torch.as_tensor(neg_ids, dtype=torch.long, device=self.args.device)
        # [batch seq_len hidden_size]
        pos_emb = self.item_emb(pos_ids)
        neg_emb = self.item_emb(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_units)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.maxlen).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_full(self, batch):
        u, seq, seq2, mask, item_idx, items,mixed_seq,mixed_time = batch
        target_out,mixed_out,source_out_g,_= self.seq_self_attention_layer(np.array(u),np.array(seq), np.array(seq2),np.array(mixed_seq))
        seq_out = mixed_out[:, -1, :]


        if self.args.sample_eval:
            test_item_emb = self.item_emb(torch.LongTensor(np.array(item_idx)).to(self.dev))
            rating_pred = torch.matmul(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)
        else:
            test_item_emb = self.item_emb.weight[1:self.args.itemnum1 + 1, :]
            rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))

        return rating_pred




