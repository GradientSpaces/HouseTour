import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size).to(dtype=torch.float64),
            Conv1dBlock(out_channels, out_channels, kernel_size).to(dtype=torch.float64),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels).to(dtype=torch.float64),
            Rearrange('batch t -> batch t 1'),
        ).to(dtype=torch.float64)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1).to(dtype=torch.float64) \
            if inp_channels != out_channels else nn.Identity().to(dtype=torch.float64)

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        
        out = self.blocks[0](x) + self.time_mlp(t.double())
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4),
    ):
        super().__init__()

        dims = [(transition_dim + cond_dim), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1).to(dtype=torch.float64),
        )

    def forward(self, x, cond, visual_cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Bottleneck activation
        bottleneck_feats = x.clone()

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x, bottleneck_feats

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, cond_dim, num_heads=4):
        """
        Args:
            channels (int): number of channels in the feature map (for queries).
            cond_dim (int): number of channels in the conditioning (visual) features.
            num_heads (int): number of attention heads.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels).to(dtype=torch.float64)
        self.q_proj = nn.Linear(channels, channels).to(dtype=torch.float64)
        self.k_proj = nn.Linear(cond_dim, channels).to(dtype=torch.float64)
        self.v_proj = nn.Linear(cond_dim, channels).to(dtype=torch.float64)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads).to(dtype=torch.float64)
        self.out_proj = nn.Linear(channels, channels).to(dtype=torch.float64)

    def forward(self, x, cond, visual_cond, attn_mask=None):
        """
        Args:
            x: Tensor of shape [batch, channels, seq_len]
            visual_cond: Conditioning tensor of shape [batch, cond_seq, cond_dim]
            attn_mask: Optional attention mask of shape [seq_len, cond_seq] 
                       (or broadcastable to that shape), where positions to mask are marked.
        Returns:
            Tensor of shape [batch, channels, seq_len]
        """
        b, c, seq_len = x.shape
        _, cond_seq, _ = visual_cond.shape

        # Rearrange x to [batch, seq_len, channels] for normalization.
        x_reshaped = x.permute(0, 2, 1)
        x_norm = self.norm(x_reshaped)
        
        # Compute queries: shape becomes [seq_len, batch, channels]
        q = self.q_proj(x_norm).permute(1, 0, 2)
        
        # Process conditioning features.
        # cond is expected to be [batch, cond_seq, cond_dim]
        k = self.k_proj(visual_cond).permute(1, 0, 2)
        v = self.v_proj(visual_cond).permute(1, 0, 2)
        
        # Apply multi-head attention with the optional custom mask.
        # Note: The attn_mask (if provided) should have shape [seq_len, cond_seq].
        if attn_mask is None:
            # mask_fill_value = float(-1e9)
            attn_mask = torch.ones((seq_len, cond_seq), dtype=torch.bool, device=q.device)
            candidate_idxs = list(sorted(cond.keys()))
            for i in range(cond_seq):
                if i == 0:
                    start = 0
                    end = candidate_idxs[i + 1] + 1
                elif i == cond_seq - 1:
                    start = candidate_idxs[i]
                    end = seq_len
                else:
                    start = candidate_idxs[i - 1]
                    end = candidate_idxs[i + 1] + 1
                attn_mask[start:end, i] = False
        
            # if not ((attn_mask == mask_fill_value).all(dim=1).sum() == 0.0).item():
            #     for idx, row in enumerate(attn_mask):
            #         if (row == mask_fill_value).all(dim=0).sum() > 0.0:
            #             print(idx)
            #             print(row)
            #             print(candidate_idxs)
            #             raise Exception


        attn_out, _ = self.attn(query=q, key=k, value=v, attn_mask=attn_mask)
        
        # Rearrange attn_out back to [batch, seq_len, channels],
        # apply output projection, and add the original residual.
        attn_out = self.out_proj(attn_out).permute(1, 0, 2)
        out = x_reshaped + attn_out
        
        # Return to original shape: [batch, channels, seq_len]
        return out.permute(0, 2, 1)

class TemporalUNetWithCrossAttention(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4),
        num_heads=4,
    ):
        """
        Args:
            horizon (int): the temporal length (number of timesteps in the 1D signal).
            transition_dim (int): the number of channels for the input (and output) transition features.
            dim (int): base channel dimension.
            dim_mults (tuple of ints): multipliers to set the channel counts at different scales.
            num_heads (int): number of heads in the cross-attention modules.
        """
        super().__init__()
        # Build channel dimensions for each stage.
        # Note: unlike some UNet variants, here the conditioning is injected via cross-attention
        # and not concatenated onto the input.
        dims = [(transition_dim + cond_dim), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # A simple time embedding MLP.
        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # Visual Features CNNs
        in_channels, mid_channels, out_channels = 1280, 128, 16
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 720, in_channels, dtype=torch.float64)
        )
        self.visual_linear = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.Mish(),
            nn.Linear(mid_channels, mid_channels),
            nn.Mish(),
            nn.Linear(mid_channels, mid_channels),
            nn.Mish(),
            nn.Linear(mid_channels, out_channels)
        ).to(dtype=torch.float64)

        # Down-sampling pathway.
        self.downs = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            # Cross-attend to visual features after the residual blocks.
            self.downs_attn.append(CrossAttentionBlock(dim_out, out_channels * 720, num_heads=num_heads))
            if not is_last:
                horizon = horizon // 2

        # Middle blocks.
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = CrossAttentionBlock(mid_dim, out_channels * 720, num_heads=num_heads)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        # Up-sampling pathway.
        self.ups = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            self.ups_attn.append(CrossAttentionBlock(dim_in, out_channels * 720, num_heads=num_heads))
            if not is_last:
                horizon = horizon * 2

        # Final convolution to project back to the transition dimension.
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1).to(dtype=torch.float64),
        )

    def forward(self, x, cond, visual_cond, time):
        """
        Args:
            x: Tensor of shape [batch, horizon, transition_dim] (the 1D temporal signal)
            visual_cond: Tensor of shape [batch, cond_seq, visual_cond_dim]
                (the visual conditioning features, e.g. extracted from an image encoder)
            time: Tensor of shape [batch, *] representing the time step(s) (e.g. noise schedule timesteps)
        Returns:
            Tensor of shape [batch, horizon, transition_dim]
        """
        batch_size, n_candidates, grid_h, grid_w, feat_dim = visual_cond.shape
        # Rearrange to [batch, channels, horizon]
        x = einops.rearrange(x, 'b h t -> b t h')
        # visual_cond = einops.rearrange(visual_cond, 'b c h w f -> (b c) f h w')
        visual_cond = einops.rearrange(visual_cond, 'b c h w f -> (b c) (h w) f')
        
        t = self.time_mlp(time)
        # print(f"Visual Cond: {visual_cond.shape}")
        visual_cond = visual_cond + self.pos_embedding
        visual_cond = self.visual_linear(visual_cond)
        # print(f"Visual Cond: {visual_cond.shape}")
        # visual_cond = einops.rearrange(visual_cond, 'c f h w -> 1 c (h w f)')
        visual_cond = einops.rearrange(visual_cond, '(b c) (h w) f -> b c (h w f)', b=batch_size, c=n_candidates, h=grid_h, w=grid_w)

        hs = []

        # Down-sampling path: process and store intermediate feature maps.
        for (resnet, resnet2, downsample), attn in zip(self.downs, self.downs_attn):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x, cond, visual_cond)
            hs.append(x)
            x = downsample(x)

        # Middle part.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond, visual_cond)
        x = self.mid_block2(x, t)

        # Up-sampling path: combine with skip-connections.
        for (resnet, resnet2, upsample), attn in zip(self.ups, self.ups_attn):
            # Concatenate along the channel axis.
            x = torch.cat((x, hs.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x, cond, visual_cond)
            x = upsample(x)

        x = self.final_conv(x)
        # Rearrange back to [batch, horizon, transition_dim]
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is a tensor of shape [B] (e.g. timesteps)
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class AttentionPool(nn.Module):
    """
    Computes a weighted sum over patch tokens.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: [B, N, feature_dim]
        weights = self.fc(x)  # [B, N, 1]
        weights = F.softmax(weights, dim=1)  # normalize over patches
        pooled = (x * weights).sum(dim=1)  # [B, feature_dim]
        return pooled


class TemporalUNetWithVisualConcat(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4),
        # Note: num_heads is no longer used because we remove cross-attention.
    ):
        """
        Args:
            horizon (int): the temporal length (number of timesteps in the 1D signal).
            transition_dim (int): number of channels for the input (and output) transition features.
            cond_dim (int): dimension of additional conditioning (e.g. text) to be concatenated.
            dim (int): base channel dimension.
            dim_mults (tuple of ints): multipliers to set the channel counts at different scales.
        """
        super().__init__()

        self.in_channels, self.mid_channels, self.out_channels = 1280, 128, 16 
        self.visual_feature_dim = self.out_channels

        base_input_dim = transition_dim + cond_dim + self.visual_feature_dim
        dims = [base_input_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )


        self.pos_embedding = nn.Parameter(torch.randn(1, 720, self.in_channels, dtype=torch.float64))
        self.visual_linear = nn.Sequential(
            nn.Linear(self.in_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.out_channels)
        ).to(dtype=torch.float64)
        # Attention pooling to aggregate patch features into one global feature.
        self.visual_pool = AttentionPool(self.out_channels).to(dtype=torch.float64)
        self.dual_pool = AttentionPool(self.out_channels).to(dtype=torch.float64)

        # --- Down-sampling Pathway ---
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon // 2

        # --- Middle Blocks ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        # --- Up-sampling Pathway ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon * 2

        # --- Final Convolution ---
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1).to(dtype=torch.float64),
        )

    def forward(self, x, cond, visual_cond, time):
        """
        Args:
            x: Tensor of shape [batch, horizon, transition_dim] (the 1D temporal signal).
            cond: Tensor of shape [batch, cond_dim] representing additional conditioning.
            visual_cond: Tensor of shape [batch, n_candidates, grid_h, grid_w, feat_dim]
                (the patch-wise visual features).
            time: Tensor representing time steps.
        Returns:
            Tensor of shape [batch, horizon, transition_dim].
        """
        batch_size, n_candidates, grid_h, grid_w, feat_dim = visual_cond.shape
        # Rearrange to [batch, channels, horizon]
        x = einops.rearrange(x, 'b h t -> b t h')
        # visual_cond = einops.rearrange(visual_cond, 'b c h w f -> (b c) f h w')
        visual_cond = einops.rearrange(visual_cond, 'b c h w f -> (b c) (h w) f')

        # Add positional embeddings (assumes total patches = 720)
        visual_cond = visual_cond + self.pos_embedding
        # Reduce each patch’s dimension using the linear network
        visual_cond = self.visual_linear(visual_cond)  # [B, num_patches, out_channels]
        # Pool the patches via attention to get a single global feature per image: [B, out_channels]
        global_visual = self.visual_pool(visual_cond)
        global_visual_pairs = torch.stack((global_visual[:-1], global_visual[1:]), dim=1)
        global_visual_pairs = self.dual_pool(global_visual_pairs)
        global_visual_pairs = einops.rearrange(global_visual_pairs, 'bc f -> f bc')

        global_visual_cond = torch.zeros(batch_size, self.out_channels, x.shape[2]).to(x.device)
        cond_idxs = list(cond.keys())
        for i, (idx1, idx2) in enumerate(zip(cond_idxs[:-1], cond_idxs[1:])):
            if i==0:
                global_visual_cond[...,:, :idx2] = global_visual_pairs[..., i].unsqueeze(-1).expand(self.out_channels, idx2)
            elif i == len(cond_idxs) - 2:
                global_visual_cond[..., idx1:] = global_visual_pairs[..., i].unsqueeze(-1).expand(self.out_channels, x.shape[2]-idx1)
            else:
                global_visual_cond[..., idx1:idx2] = global_visual_pairs[..., i].unsqueeze(-1).expand(self.out_channels, idx2-idx1)
        
        x = torch.cat([x, global_visual_cond], dim=1)

        # --- Down-sampling Path ---
        hs = []
        t_emb = self.time_mlp(time)
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            hs.append(x)
            x = downsample(x)

        # --- Middle Blocks ---
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        # --- Up-sampling Path ---
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat([x, hs.pop()], dim=1)  # concatenate along channel axis
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            x = upsample(x)

        x = self.final_conv(x)
        # Rearrange back to [batch, horizon, transition_dim]
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class TemporalUNetWithCrossAttentionPooling(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4),
        num_heads=4,
    ):
        super().__init__()

        # Build channel dimensions.
        dims = [transition_dim + cond_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim  # for time embedding

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # --- Visual Conditioning Branch ---
        # These parameters follow your previous visual branch examples.
        self.in_channels = 1280
        self.mid_channels = 128
        self.out_channels = 16
        self.num_patches = 720  # assumed total number of visual tokens/patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.in_channels, dtype=torch.float64)
        )
        self.visual_linear = nn.Sequential(
            nn.Linear(self.in_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.mid_channels),
            nn.Mish(),
            nn.Linear(self.mid_channels, self.out_channels)
        ).to(dtype=torch.float64)
        self.visual_pool = AttentionPool(self.out_channels).to(dtype=torch.float64)

        # --- Down-sampling Path ---
        self.downs = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            # Use your existing CrossAttentionBlock.
            self.downs_attn.append(
                CrossAttentionBlock(dim_out, cond_dim=self.out_channels, num_heads=num_heads)
            )
            if not is_last:
                horizon = horizon // 2

        # --- Middle Blocks ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = CrossAttentionBlock(mid_dim, cond_dim=self.out_channels, num_heads=num_heads)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        # --- Up-sampling Path ---
        self.ups = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = (ind >= (num_resolutions - 1))
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            self.ups_attn.append(
                CrossAttentionBlock(dim_in, cond_dim=self.out_channels, num_heads=num_heads)
            )
            if not is_last:
                horizon = horizon * 2

        # --- Final Convolution ---
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1).to(dtype=torch.float64),
        )

    def forward(self, x, cond, visual_cond, time):
        """
        Args:
            x: Tensor of shape [batch, horizon, transition_dim] (the 1D temporal signal).
            cond: (Not used for attention pooling; provided for API consistency.)
            visual_cond: Tensor of shape [batch, n_candidates, grid_h, grid_w, feat_dim]
                         (patch–wise visual features).
            time: Tensor of shape [batch, ...] (time step information).
        Returns:
            Tensor of shape [batch, horizon, transition_dim].
        """
        # Rearrange temporal input to [B, channels, seq_len].
        x = einops.rearrange(x, 'b h t -> b t h')

        # Process visual conditioning:
        # Rearrange from [B, n_candidates, grid_h, grid_w, feat_dim] to [B, num_patches, feat_dim].
        batch_size, n_candidates, grid_h, grid_w, feat_dim = visual_cond.shape
        visual_cond = einops.rearrange(visual_cond, 'b c h w f -> (b c) (h w) f')
        if visual_cond.shape[1] != self.num_patches:
            raise ValueError(f'Expected {self.num_patches} patches but got {visual_cond.shape[1]}')
        # Add positional embeddings and reduce dimensionality.
        visual_cond = visual_cond + self.pos_embedding
        visual_cond = self.visual_linear(visual_cond)  # [B, num_patches, out_channels]
        # Pool the visual features into one global vector per sample.
        pooled_visual = self.visual_pool(visual_cond)  # [B, out_channels]
        # For cross-attention, expand pooled visual to a singleton sequence.
        pooled_visual_exp = einops.rearrange(pooled_visual, '(b c) f -> b c f', b=batch_size)

        # Compute time embedding.
        t = self.time_mlp(time)


        hs = []
        # Down-sampling path.
        for (resnet, resnet2, downsample), attn_block in zip(self.downs, self.downs_attn):
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn_block(x, cond, pooled_visual_exp)
            hs.append(x)
            x = downsample(x)

        # Middle blocks.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, cond, pooled_visual_exp)
        x = self.mid_block2(x, t)

        # Up-sampling path.
        for (resnet, resnet2, upsample), attn_block in zip(self.ups, self.ups_attn):
            x = torch.cat((x, hs.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn_block(x, cond, pooled_visual_exp)
            x = upsample(x)

        x = self.final_conv(x)
        # Rearrange back to [B, horizon, transition_dim].
        x = einops.rearrange(x, 'b t h -> b h t')
        return x