# -*- coding: utf-8 -*-
# @Time   : 2020/10/4 16:55
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
GCSAN
################################################

Reference:
    Chengfeng Xu et al. "Graph Contextualized Self-Attention Network for Session-based Recommendation." in IJCAI 2019.

"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import EmbLoss, BPRLoss


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via gated graph neural network.

        Args:
            A (torch.FloatTensor): The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden (torch.FloatTensor): The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden))
        input_out = torch.matmul(
            A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class GCSAN(nn.Module):
    r"""GCSAN captures rich local dependencies via graph neural network,
     and learns long-range dependencies by applying the self-attention mechanism.

    Note:

        In the original paper, the attention mechanism in the self-attention layer is a single head,
        for the reusability of the project code, we use a unified transformer component.
        According to the experimental results, we only applied regularization to embedding.
    """

    def __init__(self, args):
        super(GCSAN, self).__init__()
        self.args = args
        # load parameters info
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.hidden_units = args.hidden_units  # same as embedding_size
        self.dropout_rate = args.dropout_rate
        self.item_num = args.itemnum1

        self.step = args.step
        self.device = args.device
        self.weight = args.w1
        self.reg_weight = args.reg_weight

        # define layers and loss
        self.item_emb = nn.Embedding(
            self.item_num + 1, self.hidden_units, padding_idx=0
        )
        self.gnn = GNN(self.hidden_units, self.step)
        
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_units,
            inner_size=self.hidden_units,
            hidden_dropout_prob=self.dropout_rate,
            attn_dropout_prob=self.dropout_rate,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
        )
        self.reg_loss = EmbLoss()
       

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_slice(self, item_seq):
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()

        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, item_seq):
        assert 0 <= self.weight <= 1
        item_seq = torch.LongTensor(item_seq).to(self.device)
        alias_inputs, A, items = self._get_slice(item_seq)
        hidden = self.item_emb(items)
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.hidden_units
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        ht = seq_hidden
        a = seq_hidden
        attention_mask = self.get_attention_mask(item_seq)

        outputs = self.self_attention(a, attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        
        seq_output = self.weight * output + (1 - self.weight) * ht
        return seq_output

    
    def calculate_loss(self, batch):
        _, seq, pos, neg, _, _,_,_,_ = batch
        seq_out = self.forward(np.array(seq))
      
        pos = torch.LongTensor(pos).to(self.args.device)
        test_item_emb = self.item_emb.weight[1:self.item_num + 1, :]

        logits = seq_out @ test_item_emb.T
        indices = torch.where(pos.view(-1) != 0)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], (pos.view(-1)[indices]-1), reduction='mean')
        
        return loss
    

    def predict_full(self, batch):
        # u, seq, seq2, mask, sampled_items, items, mark_items = batch
        seq_out = self.forward(np.array(batch[1]))[:, -1, :]

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
