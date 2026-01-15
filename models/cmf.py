
r"""
CMF
################################################
Reference:
    Ajit P. Singh et al. "Relational Learning via Collective Matrix Factorization." in SIGKDD 2008.
"""

import torch
import torch.nn as nn
import numpy as np


class CMF(nn.Module):
    """CMF adapted to this repository's training/eval interface."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.user_num = args.usernum
        self.item_num = args.itemnum
        self.target_item_num = args.itemnum1
        self.source_item_min = args.itemnum1 + 1
        self.source_item_max = args.itemnum1 + args.itemnum2

        self.embedding_size = getattr(args, "embedding_size", args.hidden_units)
        self.alpha = getattr(args, "alpha", 0.5)
        self.lamda = getattr(args, "lamda", 0.0)
        self.gamma = getattr(args, "gamma", 0.0)

        self.user_emb = nn.Embedding(self.user_num + 1, self.embedding_size)
        self.item_emb = nn.Embedding(self.item_num + 1, self.embedding_size, padding_idx=0)

        self.bce = nn.BCEWithLogitsLoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

    def forward(self, users, items):
        user_e = self.user_emb(users)
        item_e = self.item_emb(items)
        return (user_e * item_e).sum(dim=-1)

    def _pointwise_loss(self, users, pos_items, neg_items):
        pos_logits = self.forward(users, pos_items)
        neg_logits = self.forward(users, neg_items)

        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        pos_loss = self.bce(pos_logits, pos_labels)
        neg_loss = self.bce(neg_logits, neg_labels)
        return pos_loss + neg_loss

    def _l2_reg(self, users, items):
        if users.numel() == 0 or items.numel() == 0:
            return torch.tensor(0.0, device=users.device)
        user_e = self.user_emb(users)
        item_e = self.item_emb(items)
        return (user_e.pow(2).mean() + item_e.pow(2).mean())

    def calculate_loss(self, batch):
        user = torch.LongTensor(np.array(batch[0])).to(self.args.device)
        seq = torch.LongTensor(np.array(batch[1])).to(self.args.device)
        pos = torch.LongTensor(np.array(batch[2])).to(self.args.device)
        neg = torch.LongTensor(np.array(batch[3])).to(self.args.device)
        seq2 = torch.LongTensor(np.array(batch[4])).to(self.args.device)

        user_pos = user.unsqueeze(1).expand_as(pos)
        mask = pos > 0
        pos_users = user_pos[mask]
        pos_items = pos[mask]
        neg_items = neg[mask]

        target_loss = self._pointwise_loss(pos_users, pos_items, neg_items)
        target_reg = self._l2_reg(pos_users, pos_items) * self.gamma

        source_mask = seq2 > 0
        source_users = user.unsqueeze(1).expand_as(seq2)[source_mask]
        source_items = seq2[source_mask]

        if source_items.numel() > 0 and self.source_item_max >= self.source_item_min:
            neg_source = torch.randint(
                low=self.source_item_min,
                high=self.source_item_max + 1,
                size=source_items.shape,
                device=source_items.device,
            )
            neg_source = torch.where(neg_source == source_items, neg_source + 1, neg_source)
            neg_source = torch.where(
                neg_source > self.source_item_max,
                torch.full_like(neg_source, self.source_item_min),
                neg_source,
            )
            source_loss = self._pointwise_loss(source_users, source_items, neg_source)
            source_reg = self._l2_reg(source_users, source_items) * self.lamda
        else:
            source_loss = torch.tensor(0.0, device=self.args.device)
            source_reg = torch.tensor(0.0, device=self.args.device)

        return self.alpha * (source_loss + source_reg) + (1 - self.alpha) * (target_loss + target_reg)

    def predict_full(self, batch):
        user = torch.LongTensor(np.array(batch[0])).to(self.args.device)
        user_e = self.user_emb(user)

        if self.args.sample_eval:
            item_idx = torch.LongTensor(np.array(batch[4])).to(self.args.device)
            item_e = self.item_emb(item_idx)
            rating_pred = torch.matmul(item_e, user_e.unsqueeze(-1)).squeeze(-1)
        else:
            item_e = self.item_emb.weight[1:self.target_item_num + 1]
            rating_pred = torch.matmul(user_e, item_e.transpose(0, 1))

        return rating_pred