import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, real_value_mask):
        binarized_mask = (real_value_mask >= 0).float().requires_grad_(True)
        ctx.save_for_backward(binarized_mask.clone())
        return binarized_mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class KernelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_factor,
                 nb_random_features=10, weight=True):
        super(KernelAttention, self).__init__()
        if weight:
            self.Wk = nn.ModuleList()
            self.Wq = nn.ModuleList()
            self.Wv = nn.ModuleList()
            for _ in range(num_factor):
                self.Wk.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wq.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))
                self.Wv.append(nn.Linear(in_channels // num_factor, out_channels // num_factor))

        self.out_channels = out_channels
        self.num_factor = num_factor
        self.nb_random_features = nb_random_features
        self.weight = weight

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z, tau):
        query, key, value = torch.zeros_like(z, device=z.device), torch.zeros_like(z,
                                                                                   device=z.device), torch.zeros_like(z,
                                                                                                                      device=z.device)
        for head in range(self.num_factor):
            query[:, head] = self.Wq[head](z[:, head])
            key[:, head] = self.Wk[head](z[:, head])
            value[:, head] = self.Wv[head](z[:, head])

        dim = query.shape[-1]
        projection_matrix = create_projection_matrix(self.nb_random_features, dim).to(query.device)
        z_next = kernelized_softmax(query, key, value, projection_matrix, tau)

        z_next = z_next.flatten(-2, -1)
        return z_next.squeeze()


def create_projection_matrix(m, d):
    nb_full_blocks = int(m / d)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((d, d))
        q, _ = torch.linalg.qr(unstructured_block)
        block_list.append(q.T)
    final_matrix = torch.vstack(block_list)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix)


def softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[0] ** -0.5)

    data_dash = torch.einsum("nhd,md->nhm", (data_normalizer * data), projection_matrix)  # perform projection
    diag_data = (data ** 2).sum(-1)

    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(-1)
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps
        )
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps
        )
    return data_dash


def kernelized_softmax(query, key, value, projection_matrix=None, tau=0.5):
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    query_kernel = softmax_kernel(query, True, projection_matrix)
    key_kernel = softmax_kernel(key, False, projection_matrix)

    kvs = torch.einsum("nhm,nhd->hmd", key_kernel, value)
    numerator = torch.einsum("nhm,hmd->nhd", query_kernel, kvs)
    denominator = (query_kernel * key_kernel.sum(0, keepdim=True)).sum(-1, keepdim=True)

    z_output = numerator / denominator
    return z_output


class MFGSLAE(nn.Module):
    """MFGSLAE adapted to this repository's training/eval interface."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.target_num_items = args.itemnum1
        self.source_num_items = args.itemnum2
        self.total_num_items = args.itemnum1 + args.itemnum2

        self.lat_dim = getattr(args, "hidden_units", getattr(args, "latent_dimension", 128))
        self.drop_out = getattr(args, "dropout_rate", getattr(args, "dropout_prob", 0.1))
        self.tau = getattr(args, "tau", 0.5)
        self.factor = getattr(args, "factor", 4)
        self.epsilon = getattr(args, "epsilon", 1e-3)
        self.alpha = getattr(args, "alpha", 0.5)
        self.l1_rate = getattr(args, "l1_rate", 1e-4)
        self.ratio = getattr(args, "ratio", 0.5)
        self.ratio_threshold = getattr(args, "ratio_threshold", 0.5)

        self.source_weight = getattr(args, "source_weight", 0.1)
        self.target_weight = getattr(args, "target_weight", 0.9)
        self.compact_weight = getattr(args, "compact_weight", 0.5)
        self.l1_weight = getattr(args, "l1_weight", 0.2)

        self.hidden_dim = self.lat_dim
        self.source_encoder = nn.Linear(self.source_num_items, self.hidden_dim)
        self.target_encoder = nn.Linear(self.target_num_items, self.hidden_dim)
        self.gsl_encoder = nn.Linear(self.total_num_items, self.hidden_dim)
        self.act = nn.LeakyReLU()

        self.graph_layer = KernelAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.factor,
            nb_random_features=64
        )
        self.subspace_projector = nn.ModuleList()
        for _ in range(self.factor):
            self.subspace_projector.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // self.factor),
                    nn.ReLU(),
                )
            )
        self.norm = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ])
        self.source_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.target_aug_norm = nn.LayerNorm(self.hidden_dim)
        self.proj_source = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.proj_target = nn.Linear(self.lat_dim, self.lat_dim, bias=False)
        self.mask_source = nn.Parameter(torch.empty(1, self.factor))
        self.mask_target = nn.Parameter(torch.empty(1, self.factor))
        self.decoder_source = nn.Linear(self.lat_dim, self.source_num_items)
        self.decoder_target = nn.Linear(self.lat_dim, self.target_num_items)
        self.binary = Binarize.apply
        self.item_emb = nn.Embedding(self.target_num_items + 1, self.lat_dim, padding_idx=0)

        self.reset_parameters()
        nn.init.normal_(self.mask_source, 0, 0.01)
        nn.init.normal_(self.mask_target, 0, 0.01)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.item_emb.weight)

    def compactness(self, emb):
        emb = F.normalize(emb, dim=-1)
        loss = 0
        emb_all = torch.flatten(emb, 0, 1)  # [NK, D]
        NK, D = emb_all.shape
        loss -= 0.5 * torch.logdet(torch.eye(D, device=self.device) + (D / NK / self.epsilon * emb_all.T) @ emb_all)

        N, K, D = emb.shape
        emb = torch.einsum('abc->bca', emb)  # [K, D, N]
        loss_factor = torch.einsum('abc,adc->abd', D / N / self.epsilon * emb, emb)  # [K, D, D]
        loss_factor = 0.5 * torch.logdet(torch.eye(D, device=self.device).unsqueeze(0) + loss_factor)
        loss += loss_factor.sum()

        return loss

    def _dense_drop(self, x: torch.Tensor):
        if not self.training or self.drop_out <= 0:
            return x
        keep = torch.rand_like(x) > self.drop_out
        return x * keep / (1 - self.drop_out)

    def _build_rating_vectors(self, seq: torch.Tensor, seq2: torch.Tensor):
        batch_size = seq.size(0)
        target_r = torch.zeros(batch_size, self.target_num_items, device=seq.device)
        source_r = torch.zeros(batch_size, self.source_num_items, device=seq.device)
        combined_r = torch.zeros(batch_size, self.total_num_items, device=seq.device)

        row_idx = torch.arange(batch_size, device=seq.device).unsqueeze(1).expand_as(seq)

        target_mask = (seq > 0) & (seq <= self.target_num_items)
        if target_mask.any():
            target_cols = (seq[target_mask] - 1).long()
            target_r[row_idx[target_mask], target_cols] = 1.0
            combined_r[row_idx[target_mask], target_cols] = 1.0

        source_mask = (seq2 > self.target_num_items) & (seq2 <= self.total_num_items)
        if source_mask.any():
            source_cols = (seq2[source_mask] - self.target_num_items - 1).long()
            source_r[row_idx[source_mask], source_cols] = 1.0
            combined_cols = (seq2[source_mask] - 1).long()
            combined_r[row_idx[source_mask], combined_cols] = 1.0

        return target_r, source_r, combined_r

    def factor_gsl(self, h):
        h = self.norm[0](h)
        self.h_factor = self.factor_division(h)
        h_enhanced = self.graph_layer(self.h_factor, self.tau)
        h = h + h_enhanced
        h = self.act(h)
        h = self.norm[1](h)
        return h

    def factor_division(self, h):
        h_factor_list = []
        for idx in range(self.factor):
            h_factor = self.subspace_projector[idx](h)
            h_factor_list.append(h_factor)
        h_factor = torch.stack(h_factor_list, dim=1)
        return h_factor

    def factor_selection(self, h):
        h_source = self.proj_source(h).reshape(h.shape[0], self.factor, -1)
        h_target = self.proj_target(h).reshape(h.shape[0], self.factor, -1)
        mask_source = self.binary(self.mask_source).unsqueeze(-1)
        mask_target = self.binary(self.mask_target).unsqueeze(-1)
        h_source_aug = self.source_aug_norm((h_source * mask_source).flatten(-2, -1))
        h_target_aug = self.target_aug_norm((h_target * mask_target).flatten(-2, -1))
        return h_source_aug, h_target_aug

    def forward(self, seq: torch.Tensor, seq2: torch.Tensor):
        target_r, source_r, combined_r = self._build_rating_vectors(seq, seq2)

        source_input = self._dense_drop(F.normalize(source_r, dim=1))
        target_input = self._dense_drop(F.normalize(target_r, dim=1))
        gsl_input = self._dense_drop(F.normalize(combined_r, dim=1))

        h_source = self.act(self.source_encoder(source_input))
        h_target = self.act(self.target_encoder(target_input))

        h_gsl = self.act(self.gsl_encoder(gsl_input))
        h_gsl = self.factor_gsl(h_gsl)
        self.h_gsl = h_gsl.reshape(h_gsl.shape[0], self.factor, -1)

        if self.factor > 1:
            h_source_aug, h_target_aug = self.factor_selection(h_gsl)
        else:
            h_source_aug, h_target_aug = self.proj_source(h_gsl), self.proj_target(h_gsl)

        h_source = self.ratio * h_source + (1 - self.ratio) * h_source_aug
        h_target = self.ratio * h_target + (1 - self.ratio) * h_target_aug

        reconstructed_source = h_source @ self.decoder_source.weight.T + self.decoder_source.bias
        reconstructed_target = h_target @ self.decoder_target.weight.T + self.decoder_target.bias

        return reconstructed_source, reconstructed_target

    def epoch_start(self):
        if self.ratio > self.ratio_threshold:
            self.ratio = self.ratio * self.ratio
        else:
            self.ratio = self.ratio_threshold

    def calculate_loss(self, batch):
        seq = torch.LongTensor(np.array(batch[1])).to(self.device)
        seq2 = torch.LongTensor(np.array(batch[4])).to(self.device)

        target_r, source_r, _ = self._build_rating_vectors(seq, seq2)
        reconstructed_source, reconstructed_target = self.forward(seq, seq2)

        ce_loss_source = -(F.log_softmax(reconstructed_source, 1) * source_r).sum(1).mean()
        ce_loss_target = -(F.log_softmax(reconstructed_target, 1) * target_r).sum(1).mean()

        if self.factor > 1:
            compactness_loss = self.alpha * (self.compactness(self.h_factor) + self.compactness(self.h_gsl))
        else:
            compactness_loss = torch.tensor(0.0, device=self.device)

        l1_loss = self.l1_rate * (
            F.l1_loss(self.mask_source, torch.zeros_like(self.mask_source, device=self.device)) +
            F.l1_loss(self.mask_target, torch.zeros_like(self.mask_target, device=self.device))
        )

        total = (
            self.source_weight * ce_loss_source +
            self.target_weight * ce_loss_target +
            self.compact_weight * compactness_loss +
            self.l1_weight * l1_loss
        )
        return total

    def predict_full(self, batch):
        seq = torch.LongTensor(np.array(batch[1])).to(self.device)
        seq2 = torch.LongTensor(np.array(batch[2])).to(self.device)
        _, reconstructed_target = self.forward(seq, seq2)

        if self.args.sample_eval:
            item_idx = torch.LongTensor(np.array(batch[4])).to(self.device)
            gather_idx = item_idx - 1
            return torch.gather(reconstructed_target, 1, gather_idx)

        return reconstructed_target