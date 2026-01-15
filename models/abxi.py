from typing import Tuple, List
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



# from models.utils.position import get_absolute_pos_idx
#
# from models.data.evaluation import cal_norm_mask
# from models.utils.initialization import init_weights

class FeedForward(nn.Module):
    """
    SwiGLU-FFN
    Require manual Post-norm residual connection.
    """
    def __init__(self,
                 d_embed: int,
                 d_ffn: int = 680,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_ffn = d_ffn

        self.fc_1 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_embed, bias=False)

    def forward(self,
                h: torch.Tensor,
                ) -> torch.Tensor:
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class MultiHeadAttention(nn.Module):
    """
    Post-norm residual connection integrated.
    """
    def __init__(self,
                 args: argparse,
                 ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(args.hidden_units, args.num_heads, batch_first=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.norm_mha = nn.LayerNorm(args.hidden_units)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((args.maxlen,
                                                    args.maxlen), True), diagonal=1),
                             persistent=False)

    def forward(self,
                h: torch.Tensor,
                mask: torch.Tensor,
                )-> torch.Tensor:
        h_mha = self.norm_mha(h + self.dropout(self.mha(h, h, h,
                                                    attn_mask=self.mask_causal,
                                                    is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_mha

class LoRA(nn.Module):
    def __init__(self,
                 d_embed: torch.Tensor,
                 rank: int = 16,
                 ) -> None:
        super().__init__()
        self.mat_A = nn.Parameter(torch.randn(d_embed, rank) / 50)  # match the scale in initialization.py
        self.mat_B = nn.Parameter(torch.zeros(rank, d_embed))

    def forward(self,
                h: torch.Tensor,
                ) -> torch.Tensor:
        h = F.linear(h, self.mat_A @ self.mat_B)
        return h

def get_absolute_pos_idx(mask: torch.DoubleTensor,
                         ) -> torch.LongTensor:
    """
    Generate position index, default ignoring padding and masking index 0.
    Input mask is non-padded mask
    """
    mask = mask.long().squeeze(-1)
    return mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) * mask
def cal_norm_mask(mask: torch.Tensor,
                  ) -> torch.Tensor:
    """ calculate normalized mask """
    return mask * mask.sum(1).reciprocal().unsqueeze(-1).nan_to_num(posinf=0.0)

def init_weights(model: nn.Module,
                 ) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        elif isinstance(m, nn.Parameter):
            pass  # since only lora use nn.Parameter and it has its own initialization

        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02, a=-0.04, b=0.04)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)
def get_spe_seq(n_item_a: int,
                seq: np.array,
                gt: np.array,
                ) -> Tuple[np.array, np.array]:
    """
    Through task-guided alignment.
    """
    # 输入的是混合序列
    A = seq[(0 < seq) & (seq <= n_item_a)]  # 分解出A域的序列（中间没有pad）
    B = seq[seq > n_item_a]  # B域的序列


    seq_a = np.logical_and(0 < gt, gt <= n_item_a).astype(int)  # 预测目标gt在A域的位置 [1,0,0,1,……]
    seq_b = (gt > n_item_a).astype(int)

    # remove first interaction
    if seq[0] > n_item_a:  # B
        # seq_a[seq_a.nonzero()[0][0]] = 0
        nonzero_idx = seq_a.nonzero()[0]
        if nonzero_idx.size > 0:
            seq_a[nonzero_idx[0]] = 0
    else:
        # seq_b[seq_b.nonzero()[0][0]] = 0
        nonzero_idx = seq_b.nonzero()[0]
        if nonzero_idx.size > 0:
            seq_b[nonzero_idx[0]] = 0


    # remove last interaction
    if gt[-1] > n_item_a:  # B
        A = A[:-1]
    else:
        B = B[:-1]


    seq_a[seq_a != 0] = A
    seq_b[seq_b != 0] = B


    return seq_a, seq_b

def trim_seq(seq: np.array,
             len_trim: int,
             ) -> np.array:
    """ pad sequences to required length """
    return np.concatenate((np.zeros(max(0, len_trim - len(seq)), dtype=np.int32), seq))[-len_trim:]



def process_train(n_item_a: int,
                  len_trim: int,
                  seq_raw: List[int],
                  n_item: int,
                  n_neg: int,
                  ) -> Tuple[np.array, np.array, np.array, np.array, List]:

    """ process training sequences """

    # def get_neg(
    #         gt: np.array,
    #         cand_a: np.array,
    #         cand_b: np.array,
    #         n_neg: int,
    # ) -> np.array:
    #     n_neg_x2 = n_neg * 2
    #     gt_neg = np.zeros((len_trim, n_neg_x2))
    #
    #     for i, x in enumerate(gt):
    #         if x != 0:
    #             gt_neg[i] = np.concatenate((np.random.default_rng().choice(cand_a, size=n_neg, replace=False),
    #                                         np.random.default_rng().choice(cand_b, size=n_neg, replace=False)))
    #
    #     return gt_neg
    #（256，99）
    # seq_x, gt = seq_raw[:,:-1], seq_raw[:,1:]
    def get_neg(gt: np.array, cand_a: np.array, cand_b: np.array, n_neg: int) -> np.array:
        n_neg_x2 = n_neg * 2
        len_trim = gt.shape[0]
        gt_neg = np.zeros((len_trim, n_neg_x2), dtype=int)

        # 找出非零位置
        valid_idx = np.where(gt != 0)[0]
        num_valid = valid_idx.size

        if num_valid > 0:
            rng = np.random.default_rng()
            # 批量采样
            neg_a = rng.choice(cand_a, size=(num_valid, n_neg), replace=False)
            neg_b = rng.choice(cand_b, size=(num_valid, n_neg), replace=False)
            gt_neg[valid_idx] = np.concatenate([neg_a, neg_b], axis=1)

        return gt_neg
    idx_all_a = np.arange(1, n_item_a + 1)
    idx_all_b = np.arange(n_item_a, n_item + 1)
    seq_x=[]
    gt=[]
    seq_a=[]
    seq_b=[]
    gt_neg=[]
    for i in range(seq_raw.shape[0]):
        seq_raw_i=seq_raw[i][seq_raw[i] != 0]
        seq_x_i, gt_i = seq_raw_i[:-1], seq_raw_i[ 1:]
        seq_a_i, seq_b_i = get_spe_seq(n_item_a, seq_x_i, gt_i)

        seq_a.append(trim_seq(seq_a_i, len_trim))
        seq_b.append(trim_seq(seq_b_i, len_trim))
        seq_x.append(trim_seq(seq_x_i, len_trim))
        gt.append(trim_seq(gt_i, len_trim))

        cand_a = idx_all_a[~np.isin(idx_all_a, seq_raw_i[seq_raw_i <= n_item_a], assume_unique=True)]
        cand_b = idx_all_b[~np.isin(idx_all_b, seq_raw_i[seq_raw_i > n_item_a], assume_unique=True)]

        gt_neg_i = get_neg(gt[i], cand_a, cand_b,n_neg)


        gt_neg.append(gt_neg_i)

    # seq_a, seq_b = get_spe_seq_batch(n_item_a, seq_x, gt)


    # return *(trim_seq(x, len_trim) for x in (seq_x, seq_a, seq_b, gt)), seq_raw
    return seq_x,seq_a,seq_b,gt,gt_neg
def process_evaluate(len_trim: int,
                     seq_raw: List[int],
                     ) -> Tuple[np.array, np.array, np.array, List]:
    """ process evaluation sequences """


    seq_x=[]
    gt=[]
    for i in range(seq_raw.shape[0]):
        seq_raw_i=seq_raw[i][seq_raw[i] != 0]
        seq_x_i, gt_i = seq_raw_i[:-1], seq_raw_i[ -1:]
        seq_x.append(trim_seq(seq_x_i, len_trim))
        gt.append(gt_i)
    # seq_a, seq_b = get_spe_seq_batch(n_item_a, seq_x, gt)


    # return *(trim_seq(x, len_trim) for x in (seq_x, seq_a, seq_b, gt)), seq_raw
    return seq_x,gt,seq_raw

    return trim_seq(seq, len_trim), gt, seq_raw

class ABXI(nn.Module):
    def __init__(self,
                 args: argparse,
                 ) -> None:
        super().__init__()
        self.args = args
        self.dev = args.device
        self.bs = args.batch_size
        self.len_trim = args.maxlen
        self.n_item = args.itemnum
        self.n_item_a = args.itemnum1
        self.n_item_b = args.itemnum2
        self.n_neg = args.n_neg
        self.n_neg_x2 = args.n_neg * 2
        self.temp = args.temp
        self.dropout = nn.Dropout(p=args.dropout_rate) if args.dropout_rate  > 0. else nn.Identity()

        self.d_embed = args.hidden_units
        self.rd = args.rd
        self.ri = args.ri


        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)
        self.item_emb=self.ei

        # encoder, dlora
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(self.d_embed)

        self.dlora_x = LoRA(self.d_embed, self.rd)
        self.dlora_a = LoRA(self.d_embed, self.rd)
        self.dlora_b = LoRA(self.d_embed, self.rd)

        self.norm_sa_x = nn.LayerNorm(self.d_embed)
        self.norm_sa_a = nn.LayerNorm(self.d_embed)
        self.norm_sa_b = nn.LayerNorm(self.d_embed)

        # ilora
        self.ilora_a = LoRA(self.d_embed, self.ri)
        self.ilora_b = LoRA(self.d_embed, self.ri)

        # proj
        self.proj_i = FeedForward(self.d_embed)
        self.proj_a = FeedForward(self.d_embed)
        self.proj_b = FeedForward(self.d_embed)

        self.norm_i2a = nn.LayerNorm(self.d_embed)
        self.norm_i2b = nn.LayerNorm(self.d_embed)
        self.norm_a2a = nn.LayerNorm(self.d_embed)
        self.norm_b2b = nn.LayerNorm(self.d_embed)

        self.idx_all_a = np.arange(1, self.n_item_a + 1)
        self.idx_all_b = np.arange(self.n_item_a, self.n_item + 1)

        self.apply(init_weights)

    def embed_pos(self,
                  mask: torch.Tensor,
                  ) -> torch.Tensor:
        return self.ep(get_absolute_pos_idx(mask))

    def forward(self,
                seq_x: torch.Tensor,
                seq_a: torch.Tensor,
                seq_b: torch.Tensor,
                mask_gt_a: torch.Tensor,
                mask_gt_b: torch.Tensor,
                ) -> torch.Tensor:
        # masking
        mask_x = torch.where(seq_x != 0, 1., 0.).unsqueeze(-1)
        mask_a = torch.where(seq_a != 0, 1., 0.).unsqueeze(-1)
        mask_b = torch.where(seq_b != 0, 1., 0.).unsqueeze(-1)

        # embedding
        h_x = (self.ei(seq_x) + self.embed_pos(mask_x)) * mask_x
        h_a = (self.ei(seq_a) + self.embed_pos(mask_a)) * mask_a
        h_b = (self.ei(seq_b) + self.embed_pos(mask_b)) * mask_b

        h_x = self.dropout(h_x)
        h_a = self.dropout(h_a)
        h_b = self.dropout(h_b)

        # mha
        h_x = self.mha(h_x, mask_x)
        h_a = self.mha(h_a, mask_a)
        h_b = self.mha(h_b, mask_b)

        # switch training / evaluating
        if self.training:
            mask_gt_a = mask_gt_a.unsqueeze(-1)
            mask_gt_b = mask_gt_b.unsqueeze(-1)

        else:
            mask_x = mask_a = mask_b = 1
            h_x = h_x[:, -1]
            h_a = h_a[:, -1]
            h_b = h_b[:, -1]


        # ffn + dlora
        h_x = self.norm_sa_x(h_x +
                             self.dropout(self.ffn(h_x)) +
                             self.dropout(self.dlora_x(h_x))
                             ) * mask_x

        h_a = self.norm_sa_a(h_a +
                             self.dropout(self.ffn(h_a)) +
                             self.dropout(self.dlora_a(h_a))
                             ) * mask_a

        h_b = self.norm_sa_b(h_b +
                             self.dropout(self.ffn(h_b)) +
                             self.dropout(self.dlora_b(h_b))
                             ) * mask_b

        # projector + ilora
        h_i = self.proj_i(h_x)

        h_a = (self.norm_i2a((h_x +
                              self.dropout(h_i) +
                              self.dropout(self.ilora_a(h_x))
                              ) * mask_gt_a) +
               self.norm_a2a((h_a +
                              self.dropout(self.proj_a(h_a))
                              ) * mask_gt_a))

        h_b = (self.norm_i2b((h_x +
                              self.dropout(h_i) +
                              self.dropout(self.ilora_b(h_x))
                              ) * mask_gt_b) +
               self.norm_b2b((h_b +
                              self.dropout(self.proj_b(h_b))
                              ) * mask_gt_b))

        h = h_a * mask_gt_a + h_b * mask_gt_b

        return h

    def cal_rec_loss(self,
                     h: torch.Tensor,
                     gt: torch.Tensor,
                     gt_neg: torch.Tensor,
                     mask_gt_a: torch.Tensor,
                     mask_gt_b: torch.Tensor,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ InfoNCE """
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg.long())

        logits = torch.cat(((h * e_gt).unsqueeze(-2).sum(-1),
                            (h.unsqueeze(-2) * e_neg).sum(-1)), dim=-1).div(self.temp)


        loss = -F.log_softmax(logits, dim=2)[:, :, 0]
        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()
        return loss_a, loss_b

    def cal_domain_rank(self,
                        h: torch.Tensor,
                        e_gt: torch.Tensor,
                        e_mtc: torch.Tensor,
                        mask_gt_a: torch.Tensor,
                        mask_gt_b: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ calculate domain rank via inner-product similarity """
        logit_gt = (h * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logit_mtc = (h.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logit_mtc - logit_gt).gt(0).sum(-1).add(1)
        ranks_a = ranks[mask_gt_a == 1].tolist()
        ranks_b = ranks[mask_gt_b == 1].tolist()

        return ranks_a, ranks_b

    def cal_rank(self,
                 h_f: torch.Tensor,
                 gt: torch.Tensor,
                 gt_mtc: torch.Tensor,
                 mask_gt_a: torch.Tensor,
                 mask_gt_b: torch.Tensor,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ rank via inner-product similarity """
        mask_gt_a = mask_gt_a.squeeze(-1)
        mask_gt_b = mask_gt_b.squeeze(-1)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        ranks_f2a, ranks_f2b = self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a, mask_gt_b)

        return ranks_f2a, ranks_f2b





    def calculate_loss(self, batch):

        u, seq, pos, neg, seq2, _ ,mixed_seq,mixed_time,_ = batch

        # first_nonzero_pos = np.array([row[row != 0][0] if np.any(row != 0) else 0 for row in pos])
        # # 变成 shape (256, 1)
        # first_nonzero_pos = first_nonzero_pos.reshape(-1, 1)

        mask = pos != 0
        # 找到每行第一个非零元素的索引
        idx = np.argmax(mask, axis=1)
        # 如果整行都是0，则mask.sum(axis=1)==0
        first_nonzero_pos = pos[np.arange(pos.shape[0]), idx]
        first_nonzero_pos[mask.sum(axis=1) == 0] = 0
        first_nonzero_pos = first_nonzero_pos.reshape(-1, 1)

        # 拼接
        mixed_seq_new = np.concatenate([mixed_seq, first_nonzero_pos], axis=1)

        seq_x, seq_a, seq_b, gt,gt_neg=process_train(self.n_item_a,self.len_trim,np.array(mixed_seq_new),self.n_item,self.n_neg)



        seq_x, seq_a, seq_b, gt, gt_neg = [
            torch.from_numpy(np.array(x)).to(self.dev) for x in [seq_x, seq_a, seq_b, gt, gt_neg]
        ]



        mask_gt_a = torch.where(gt <= self.n_item_a, 1., 0.)
        mask_gt_b = torch.where(gt > self.n_item_a, 1., 0.)

        # mask_gt_a = mask_gt_a.to(self.dev)
        # mask_gt_b = mask_gt_b.to(self.dev)


        h = self.forward(seq_x, seq_a, seq_b, mask_gt_a, mask_gt_b)

        loss_a, loss_b = self.cal_rec_loss(h, gt, gt_neg, mask_gt_a, mask_gt_b)

        loss=loss_a+loss_b+self.CE_loss(h, pos)

        return loss
    def CE_loss(self,seq_out,pos):
        pos = torch.LongTensor(pos).to(self.args.device)
        test_item_emb = self.ei.weight[1:self.args.itemnum1 + 1, :]  # 物品编码是从1开始的，0是padding值。
        # 这里需要保留padding进行计算，不然会导致CEloss找不到对应位置。即映射概率值是从0开始，但是pos的标签是从1开始索引的。
        logits = seq_out @ test_item_emb.T
        indices = torch.where(pos.view(-1) != 0)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos.view(-1)[indices] - 1,
                               reduction='mean')
        return loss
    def predict_full(self, batch):
        u, seq, seq2, mask, item_idx, items,mixed_seq,mixed_time = batch


        # seq_x,gt,gt_mtc=process_evaluate(self.len_trim,np.array(mixed_seq))

        seq_x, gt = [
            torch.from_numpy(np.array(arr)).to(self.dev) for arr in [mixed_seq, np.array(item_idx)[:,0].reshape(-1, 1)]
        ]

        mask_gt_a = torch.where(gt <= self.n_item_a, 1., 0.).to(self.dev)
        mask_gt_b = torch.where(gt > self.n_item_a, 1., 0.).to(self.dev)

        seq_a = torch.where((seq_x > 0) & (seq_x <= self.n_item_a), seq_x, 0)
        seq_b = torch.where(seq_x > self.n_item_a, seq_x, 0)

        h = self.forward(seq_x, seq_a, seq_b, mask_gt_a, mask_gt_b)

        if self.args.sample_eval:
            item_idx = batch[4]
            # 局部评估:使用正样本+采样负样本的item embedding
            try:
                test_item_emb = self.ei(torch.LongTensor(np.array(item_idx)).to(self.dev))
                # test_item_emb = self.item_emb.weight[np.array(item_idx)]
            except Exception as e:
                print(f"Error with item_idx: {item_idx}")
                print(f"item_idx shape: {np.array(item_idx).shape}, dtype: {np.array(item_idx).dtype}")
                raise e  # 输出维度为：[batch,item_idx_num,hidden_size]
            rating_pred = torch.matmul(test_item_emb, h.unsqueeze(-1)).squeeze(-1)
            # 输出维度为：[batch,item_idx_num]
        else:
            # 全局评估:使用所有item的embedding(除padding token)
            test_item_emb = self.ei.weight[1:self.args.itemnum1 + 1, :]
            # 输出维度为：[item_num,hidden_size]
            rating_pred = torch.matmul(h,test_item_emb.transpose(0, 1))
            # 输出维度为：[batch,item_num]
        return rating_pred