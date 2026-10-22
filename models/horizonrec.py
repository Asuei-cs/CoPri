import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ============================
# PointWise FeedForward (for SASRec)
# ============================
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


# ============================
# SASRec Encoder (decoupled from item_emb)
# ============================
class SASRecEncoder(nn.Module):
    """SASRec Transformer Encoder, decoupled from item_emb, accepts pre-embedded sequences"""

    def __init__(self, hidden_units, max_len, num_blocks, num_heads, dropout_rate):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

    def forward(self, seqs_emb):
        """
        Args:
            seqs_emb: [B, L, D] Sequence with embedding already looked up
        Returns:
            log_feats: [B, L, D]
        """
        B, L, D = seqs_emb.shape
        device = seqs_emb.device

        # Before adding position embedding, determine padding based on input embedding (padding_idx=0 -> all 0s)
        timeline_mask = (seqs_emb.abs().sum(dim=-1) == 0)
        # Consistent with original SASRec: scale embedding by sqrt(D) to maintain dot product scale
        seqs_emb = seqs_emb * (D ** 0.5)

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        seqs_emb = seqs_emb + self.pos_emb(positions)
        seqs_emb = self.emb_dropout(seqs_emb)
        seqs_emb *= ~timeline_mask.unsqueeze(-1)

        attention_mask = ~torch.tril(
            torch.ones((L, L), dtype=torch.bool, device=device))

        seqs = seqs_emb
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                       attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats


# ============================
# Main Model: HorizonRec (DiffuSAS)
# ============================
class HorizonRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dev = args.device

        self.hidden_size = args.hidden_units

        # ========== Three Independent Embeddings + Encoders ==========
        # Target_emb: Target domain items (1..itemnum1)
        self.Target_item_emb = nn.Embedding(args.itemnum1 + 1, args.hidden_units, padding_idx=0)
        # Source_emb: Source domain items (1..itemnum2) — seq2 passed from framework needs to subtract itemnum1
        self.Source_item_emb = nn.Embedding(args.itemnum2 + 1, args.hidden_units, padding_idx=0)
        # Mix_emb: Target+Source domain (1..itemnum1+itemnum2) — use framework's mixed_seq for direct lookup
        self.Mix_item_emb = nn.Embedding(args.itemnum1 + args.itemnum2 + 1, args.hidden_units, padding_idx=0)

        # item_emb required by framework (for ILD and other metric calculations), points to target domain embedding
        self.item_emb = self.Target_item_emb

        # Three independent SASRec Encoders
        self.Target_Encoder = SASRecEncoder(
            args.hidden_units, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)
        self.Source_Encoder = SASRecEncoder(
            args.hidden_units, args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)
        # Framework's mixed_seq length = 2 * maxlen
        self.Mix_Encoder = SASRecEncoder(
            args.hidden_units, 2 * args.maxlen, args.num_blocks, args.num_heads, args.dropout_rate)

        # ========== Diffusion Parameters ==========
        self.timesteps = getattr(args, 'timesteps', 32)
        beta_sche = getattr(args, 'beta_sche', 'cosine')
        beta_start = getattr(args, 'beta_start', 0.001)
        beta_end = getattr(args, 'beta_end', 0.02)

        if beta_sche == 'linear':
            betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)
        elif beta_sche == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        else:
            betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                             F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef1',
                             self.betas * torch.sqrt(self.alphas_cumprod_prev) /
                             (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
                             (1. - self.alphas_cumprod))
        self.register_buffer('posterior_variance',
                             self.betas * (1. - self.alphas_cumprod_prev) /
                             (1. - self.alphas_cumprod))

        # ========== Triplet Self-Attention Fusion ==========
        self.ln = nn.LayerNorm(args.hidden_units, elementwise_affine=False)
        self.w_q = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.w_k = nn.Linear(args.hidden_units, args.hidden_units)
        self.w_v = nn.Linear(args.hidden_units, args.hidden_units)
        for m in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # ========== MDR Parameters ==========
        self.top_k = getattr(args, 'mdr_top_k', 10)
        self.w = getattr(args, 'fusion_weight', 0.6)
        self.temperature = getattr(args, 'temperature', 1.0)
        self.diffusion_loss_weight = getattr(args, 'diffusion_loss_weight', 0.1)

        # MDR user preference library (initially empty, filled on first call)
        self.register_buffer('users_mix_perferences',
                             torch.zeros(0, args.hidden_units))
        self._mdr_initialized = False

    # ==================== Diffusion Tools ====================
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_ac = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_one_minus_ac * noise

    # ==================== Time Encoding ====================
    def get_time_s(self, time):

    # ==================== MDR: Sequence Distribution Retrieval ====================
    def get_sequence_distribution(self, sequence_embed):
        """MDR: Noise prior based on Top-K retrieval"""
        if self._mdr_initialized and self.users_mix_perferences.shape[0] > 0:
            score = torch.matmul(sequence_embed, self.users_mix_perferences.t())
            _, top_score_k = torch.topk(score, k=min(self.top_k, score.shape[-1]), dim=-1)
            top_k_perference = self.users_mix_perferences[top_score_k]
            sequence_mean = torch.mean(top_k_perference, dim=1)
            sequence_var = torch.var(top_k_perference, dim=1)
            noise = torch.randn_like(sequence_embed)
            return sequence_mean + torch.sqrt(sequence_var + 1e-8) * noise
        else:
            # Use standard Gaussian noise when MDR is not initialized
            return torch.randn_like(sequence_embed)

    def _init_mdr_if_needed(self, mix_emb_last):
        """Initialize MDR preference library with first batch data"""
        if not self._mdr_initialized and self.training:
            with torch.no_grad():
                self.users_mix_perferences = mix_emb_last.detach().cpu()
                self._mdr_initialized = True

    def _update_mdr(self, mix_emb_last):
        """Online update MDR preference library (EMA)"""
        if self.training:
            new_prefs = mix_emb_last.detach().cpu()
            if self.users_mix_perferences.shape[0] == 0:
                self.users_mix_perferences = new_prefs
            else:
                # Limit MDR library max capacity to prevent memory explosion
                max_mdr_size = 20000
                self.users_mix_perferences = torch.cat(
                    [self.users_mix_perferences, new_prefs], dim=0)
                if self.users_mix_perferences.shape[0] > max_mdr_size:
                    self.users_mix_perferences = \
                        self.users_mix_perferences[-max_mdr_size:]

    # ==================== Triplet Self-Attention ====================
    def selfAttention(self, features):
        """Fuse 4 tokens: target, source, mix, time"""
        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn = q.mul(self.hidden_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        features = attn @ v
        return features[:, 0, :], features[:, 1, :], torch.mean(features, dim=1)

    # ==================== Three-way Encoding ====================
    def encode_triplet(self, seq, seq2, mixed_seq):
        """
        Encode target/source/mixed sequences in three ways

        Args:
            seq: [B, L] Target domain sequence (ID range 1..itemnum1)
            seq2: [B, L] Source domain sequence (ID range itemnum1+1..itemnum1+itemnum2)
            mixed_seq: [B, L] Mixed sequence
        Returns:
            target_feat, source_feat, mix_feat: [B, L, D]
        """
        device = seq.device

        # --- Target domain: directly use seq to look up Target_item_emb ---
        seq_safe = torch.clamp(seq, min=0, max=self.args.itemnum1)
        target_emb = self.Target_item_emb(seq_safe)
        target_log_feats = self.Target_Encoder(target_emb)

        # --- Source domain: seq2 needs to be subtracted by itemnum1 to map to 1..itemnum2 ---
        source_seq = seq2 - self.args.itemnum1
        source_seq = torch.clamp(source_seq, min=0, max=self.args.itemnum2)  # also truncate upper bound
        source_emb = self.Source_item_emb(source_seq)
        source_log_feats = self.Source_Encoder(source_emb)

        # --- Mixed domain: mixed_seq length is 2*maxlen ---
        max_mix_id = self.args.itemnum1 + self.args.itemnum2
        mixed_seq_safe = torch.clamp(mixed_seq, min=0, max=max_mix_id)
        mix_emb = self.Mix_item_emb(mixed_seq_safe)
        mix_log_feats = self.Mix_Encoder(mix_emb)

        # Return full 3D sequence representation [B, L, D] for position-wise CE use
        return target_log_feats, source_log_feats, mix_log_feats

    # ==================== Framework Interface 1: calculate_loss ====================
    def calculate_loss(self, batch, **kwargs):
        """
        batch: (u, seq, pos, neg, seq2, mask, mixed_seq, mixed_time, neg_99)
        Returns: Tensor (scalar)
        """
        u, seq, pos, neg, seq2, mask, mixed_seq, mixed_time, neg_99 = batch
        seq = torch.LongTensor(seq).to(self.dev)
        seq2 = torch.LongTensor(seq2).to(self.dev)
        pos = torch.LongTensor(pos).to(self.dev)
        mixed_seq = torch.LongTensor(mixed_seq).to(self.dev)

        # 1. Three-way encoding (3D: [B, L, D])
        target_feat, source_feat, mix_feat = self.encode_triplet(seq, seq2, mixed_seq)

        # Diffusion path still takes 2D representation at the last moment
        target_feat_last = target_feat[:, -1, :]
        source_feat_last = source_feat[:, -1, :]
        mix_feat_last = mix_feat[:, -1, :]

        # 2. Online update MDR library
        self._update_mdr(mix_feat_last)

        # 3. MDR noise prior
        target_noise = self.get_sequence_distribution(target_feat_last)
        source_noise = self.get_sequence_distribution(source_feat_last)

        # 4. Forward diffusion
        times_info = torch.randint(
            0, self.timesteps, (len(target_feat_last),), device=self.dev).long()
        next_item_noise = self.q_sample(target_feat_last, times_info, target_noise)
        source_next_item_noise = self.q_sample(source_feat_last, times_info, source_noise)
        times_info_embedding = self.get_time_s(times_info)

        # 5. Triplet Self-Attention fusion
        diffu_log_feats = torch.cat([
            next_item_noise.unsqueeze(1),
            source_next_item_noise.unsqueeze(1),
            mix_feat_last.unsqueeze(1),
            times_info_embedding.unsqueeze(1)
        ], dim=1)

        target_final_feats, source_final_feats, mean_final_feats = \
            self.selfAttention(diffu_log_feats)

        # 6. Diffusion loss (MSE)
        loss_diff = F.mse_loss(target_final_feats, target_feat_last) + \
                    F.mse_loss(source_final_feats, source_feat_last)

        # 7. Recommendation loss: position-wise CE (CD-SASRec / TJAPL style)
        # Position-wise fusion of user representation [B, L, D]
        u_p_full = (1 - self.w) * (target_feat + source_feat) + self.w * target_feat

        test_item_emb = self.Target_item_emb.weight[1:self.args.itemnum1 + 1, :]
        logits = torch.matmul(u_p_full, test_item_emb.T) * self.temperature

        # Filter padding, calculate CE for all valid positions
        pos_flat = pos.view(-1)
        indices = torch.where(pos_flat != 0)
        loss_rec = F.cross_entropy(
            logits.view(-1, logits.shape[-1])[indices],
            pos_flat[indices] - 1, reduction='mean')

        return loss_rec + self.diffusion_loss_weight * loss_diff

    # ==================== Framework Interface 2: predict_full ====================
    def predict_full(self, batch, **kwargs):
        """
        batch (eval): (u, seq, seq2, mask, item_idx, items, mixed_seq, mixed_time)
        Returns:
            rating_pred: [B, itemnum1] (full) or [B, num_sampled] (sampled)
        """
        u, seq, seq2, mask, item_idx, items, mixed_seq, mixed_time = batch[:8]
        seq = torch.LongTensor(np.array(seq)).to(self.dev)
        seq2 = torch.LongTensor(np.array(seq2)).to(self.dev)
        mixed_seq = torch.LongTensor(np.array(mixed_seq)).to(self.dev)

        # 1. Three-way encoding (3D: [B, L, D]), take 2D representation at the last moment for reverse diffusion
        target_feat, source_feat, mix_feat = self.encode_triplet(seq, seq2, mixed_seq)
        target_feat = target_feat[:, -1, :]
        source_feat = source_feat[:, -1, :]
        mix_feat = mix_feat[:, -1, :]

        # 2. MDR noise prior
        source_noise = self.get_sequence_distribution(source_feat)
        target_noise = self.get_sequence_distribution(target_feat)

        # 3. Reverse diffusion (DDPM)
        x_target = target_noise
        x_source = source_noise

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((x_target.shape[0],), i, device=self.dev, dtype=torch.long)
            times_info_emb = self.get_time_s(t)

            diffu_log_feats = torch.cat([
                x_target.unsqueeze(1),
                x_source.unsqueeze(1),
                mix_feat.unsqueeze(1),
                times_info_emb.unsqueeze(1)
            ], dim=1)

            x_start_target, x_start_source, _ = self.selfAttention(diffu_log_feats)

            model_mean_target = (
                self.extract(self.posterior_mean_coef1, t, x_target.shape) * x_start_target +
                self.extract(self.posterior_mean_coef2, t, x_target.shape) * x_target)
            model_mean_source = (
                self.extract(self.posterior_mean_coef1, t, x_source.shape) * x_start_source +
                self.extract(self.posterior_mean_coef2, t, x_source.shape) * x_source)

            if i == 0:
                x_target = model_mean_target
                x_source = model_mean_source
            else:
                posterior_variance_t = self.extract(
                    self.posterior_variance, t, x_target.shape)
                x_target = model_mean_target + \
                    torch.sqrt(posterior_variance_t) * torch.randn_like(x_target)
                x_source = model_mean_source + \
                    torch.sqrt(posterior_variance_t) * torch.randn_like(x_source)

        # 4. Final fusion
        u_p = (1 - self.w) * (x_target + x_source) + self.w * target_feat

        # 5. Scoring
        if self.args.sample_eval:
            test_item_emb = self.Target_item_emb(
                torch.LongTensor(np.array(item_idx)).to(self.dev))
            rating_pred = torch.matmul(
                test_item_emb, u_p.unsqueeze(-1)).squeeze(-1)
        else:
            test_item_emb = self.Target_item_emb.weight[1:, :]
            rating_pred = torch.matmul(u_p, test_item_emb.T)

        return rating_pred * self.temperature