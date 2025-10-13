import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from pytorch3d.transforms import quaternion_to_matrix, quaternion_to_axis_angle

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1).to(torch.float64)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1).to(torch.float64)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2).to(torch.float64),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels).to(torch.float64),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float64):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class GeodesicL2Loss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass
    
    def _loss(self, pred, targ):
        # Compute L2 loss for the first three dimensions
        l2_loss = F.mse_loss(pred[..., :3], targ[..., :3], reduction='mean')
        
        # Normalize to unit quaternions for the last four dimensions
        pred_quat = pred[..., 3:] / pred[..., 3:].norm(dim=-1, keepdim=True)
        targ_quat = targ[..., 3:] / targ[..., 3:].norm(dim=-1, keepdim=True)

        assert not torch.isnan(pred_quat).any(), "Pred Quat has NaNs"
        assert not torch.isnan(targ_quat).any(), "Targ Quat has NaNs"
        
        # Compute dot product for the quaternions
        dot_product = torch.sum(pred_quat * targ_quat, dim=-1)
        dot_product = torch.clamp(torch.abs(dot_product), -1.0, 1.0)
        
        # Compute geodesic loss for the quaternions
        geodesic_loss = 2 * torch.acos(dot_product).mean()

        assert not torch.isnan(geodesic_loss).any(), "Geodesic Loss has NaNs"
        assert not torch.isnan(l2_loss).any(), "L2 Loss has NaNs"

        return l2_loss + geodesic_loss, l2_loss, geodesic_loss

    def forward(self, pred, targ):
        loss, l2, geodesic = self._loss(pred, targ)
        
        info = {
            'l2': l2.item(),
            'geodesic': geodesic.item(),
        }
        
        return loss, info

class RotationTranslationLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass
    
    def _loss(self, pred, targ, cond=None):
        
        # Make sure the dtype is float64
        pred = pred.to(torch.float64)
        targ = targ.to(torch.float64)
        
        eps = 1e-8

        pred_trans = pred[..., :3]
        pred_quat = pred[..., 3:7]
        targ_trans = targ[..., :3]
        targ_quat = targ[..., 3:7]

        l2_loss = F.mse_loss(pred_trans, targ_trans, reduction='mean')

        # Calculate the geodesic loss
        pred_n = pred_quat.norm(dim=-1, keepdim=True).clamp(min=eps)
        targ_n = targ_quat.norm(dim=-1, keepdim=True).clamp(min=eps)
        
        pred_quat_norm = pred_quat / pred_n
        targ_quat_norm = targ_quat / targ_n


        dot_product = torch.sum(pred_quat_norm * targ_quat_norm, dim=-1).clamp(min=-1.0 + eps, max=1.0 - eps)
        quaternion_dist = 1 - (dot_product ** 2).mean()

        # Calculate the rotation error
        pred_rot = quaternion_to_matrix(pred_quat_norm).reshape(-1, 3, 3)
        targ_rot = quaternion_to_matrix(targ_quat_norm).reshape(-1, 3, 3)

        r2r1 = pred_rot @ targ_rot.permute(0, 2, 1)
        trace = torch.diagonal(r2r1, dim1=-2, dim2=-1).sum(-1)
        trace = torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps)
        geodesic_loss = torch.acos(trace).mean()

        # Add a smoothness and acceleration term to the positions and quaternions
        alpha = 1.0
        smoothness_loss = F.mse_loss(pred[:, 1:, :7].reshape(-1, 7), pred[:, :-1, :7].reshape(-1, 7), reduction='mean')
        acceleration_loss = F.mse_loss(pred[:, 2:, :7].reshape(-1, 7), 2 * pred[:, 1:-1, :7].reshape(-1, 7) - pred[:, :-2, :7].reshape(-1, 7), reduction='mean')

        l2_multiplier = 10.0

        loss = l2_multiplier * l2_loss + quaternion_dist + geodesic_loss + alpha * (smoothness_loss + acceleration_loss)

        dtw = DynamicTimeWarpingLoss()
        dtw_loss, _ = dtw.forward(pred_trans.reshape(-1, 3), targ_trans.reshape(-1, 3))

        hausdorff = HausdorffDistanceLoss()
        hausdorff_loss, _ = hausdorff.forward(pred_trans.reshape(-1, 3), targ_trans.reshape(-1, 3))

        frec = FrechetDistanceLoss()
        frechet_loss, _ = frec.forward(pred_trans.reshape(-1, 3), targ_trans.reshape(-1, 3))

        chamfer = ChamferDistanceLoss()
        chamfer_loss, _ = chamfer.forward(pred_trans.reshape(-1, 3), targ_trans.reshape(-1, 3))

        return loss, l2_loss, geodesic_loss, quaternion_dist, dtw_loss, hausdorff_loss, frechet_loss, chamfer_loss


    def forward(self, pred, targ, cond=None):
        loss, err_t, err_geo, err_r, err_dtw, err_hausdorff, err_frechet, err_chamfer = self._loss(pred, targ, cond)
        
        info = {
            'rot. error': err_r.item(),
            'geodesic error': err_geo.item(),
            'trans. error': err_t.item(),
            'dtw': err_dtw.item(),
            'hausdorff': err_hausdorff.item(),
            'frechet': err_frechet.item(),
            'chamfer': err_chamfer.item(),
        }
        
        return loss, info

class SplineLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.scales = None

    def compute_spline_coeffs(self, trans):
        p0 = trans[:, :-3, :]
        p1 = trans[:, 1:-2, :]
        p2 = trans[:, 2:-1, :]
        p3 = trans[:, 3:, :]

        # Tangent approximations
        m1 = 0.5 * (-p0 + p2)
        m2 = 0.5 * (-p1 + p3)

        # Cubic spline coefficients for each dimension
        a = (2 * p1 - 2 * p2 + m1 + m2)
        b = (-3 * p1 + 3 * p2 - 2 * m1 - m2)
        c = (m1)
        d = (p1)

        return torch.stack([a, b, c, d], dim=-1)
    
    def q_normalize(self, q):
        return q / q.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    
    def q_conjugate(self, q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([w, -x, -y, -z], dim=-1)

    def q_multiply(self, q1, q2):
        """
        q1*q2.
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=-1)

    def q_inverse(self, q):
        return self.q_conjugate(self.q_normalize(q))

    def q_log(self, q):
        """
        Quaternion logarithm for a unit quaternion
        Only returns the imaginary part
        """
        q = self.q_normalize(q)
        w = q[..., 0]
        xyz = q[..., 1:]  # shape [..., 3]
        mag_v = xyz.norm(p=2, dim=-1)
        eps = 1e-12
        angle = torch.acos(w.clamp(-1.0 + eps, 1.0 - eps))

        # We do a safe-guard against zero for sin(angle)
        small_mask = (mag_v < 1e-12) | (angle < 1e-12)
        # Where small_mask is True => near identity => log(q) ~ 0
        log_val = torch.zeros_like(xyz)

        # Normal case
        scale = angle / mag_v.clamp(min=1e-12)
        normal_case = scale.unsqueeze(-1) * xyz

        log_val = torch.where(
            small_mask.unsqueeze(-1),
            torch.zeros_like(xyz),
            normal_case
        )
        return log_val

    def q_exp(self, v):
        """
        Quaternion exponential
        """
        norm_v = v.norm(p=2, dim=-1)
        small_mask = norm_v < 1e-12

        w = torch.cos(norm_v)
        sin_v = torch.sin(norm_v)
        scale = torch.where(
            small_mask,
            torch.zeros_like(norm_v),  # if zero, sin(0)/0 => 0
            sin_v / norm_v.clamp(min=1e-12)
        )
        xyz = scale.unsqueeze(-1) * v

        # For small angles, we approximate cos(norm_v) ~ 1, sin(norm_v)/norm_v ~ 1
        w = torch.where(
            small_mask,
            torch.ones_like(w),
            w
        )
        return torch.cat([w.unsqueeze(-1), xyz], dim=-1)

    def q_slerp(self, q1, q2, t):
        """
        Spherical linear interpolation from q1 to q2 at t in [0,1].
        Both q1, q2 assumed normalized.
        q1, q2, t can be 1D or broadcastable shapes, but typically 1D.
        """
        q1 = self.q_normalize(q1)
        q2 = self.q_normalize(q2)
        dot = (q1 * q2).sum(dim=-1, keepdim=True)  # the dot product

        eps = 1e-12
        dot = dot.clamp(-1.0 + eps, 1.0 - eps)

        flip_mask = dot < 0.0
        if flip_mask.any():
            q2 = torch.where(flip_mask, -q2, q2)
            dot = torch.where(flip_mask, -dot, dot)

        # If they're very close, do a simple linear interpolation
        close_mask = dot.squeeze(-1) > 0.9995
        # Using an epsilon to avoid potential issues close to 1.0

        # Branch 1: Very close
        # linear LERP
        lerp_val = (1.0 - t) * q1 + t * q2
        lerp_val = self.q_normalize(lerp_val)

        # Branch 2: Standard SLERP
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        s1 = torch.sin(theta_0 - theta) / sin_theta_0.clamp(min=1e-12)
        s2 = torch.sin(theta) / sin_theta_0.clamp(min=1e-12)
        slerp_val = s1 * q1 + s2 * q2
        slerp_val = self.q_normalize(slerp_val)

        # Combine
        return torch.where(
            close_mask.unsqueeze(-1),
            lerp_val,
            slerp_val
        )

    def compute_uniform_tangent(self, q_im1, q_i, q_ip1):
        """
        Computes a 'Catmull–Rom-like' tangent T_i for quaternion q_i,
        given neighbors q_im1, q_i, q_ip1.

        T_i = q_i * exp( -0.25 * [ log(q_i^-1 q_ip1) + log(q_i^-1 q_im1) ] )
        """
        q_im1 = self.q_normalize(q_im1)
        q_i   = self.q_normalize(q_i)
        q_ip1 = self.q_normalize(q_ip1)

        inv_qi = self.q_inverse(q_i)
        r1 = self.q_multiply(inv_qi, q_ip1)
        r2 = self.q_multiply(inv_qi, q_im1)

        lr1 = self.q_log(r1)
        lr2 = self.q_log(r2)

        m = -0.25 * (lr1 + lr2)
        exp_m = self.q_exp(m)
        return self.q_multiply(q_i, exp_m)

    def compute_all_uniform_tangents(self, quats):
        """
        Vectorized version that computes tangents T_i for all keyframe quaternions at once.
        quats shape: [N,4], N >= 2
        Returns shape [N,4].
        """
        q_im1 = torch.cat([quats[[0]], quats[:-1]], dim=0)   # q_im1[0]   = q0
        q_ip1 = torch.cat([quats[1:], quats[[-1]]], dim=0)   # q_ip1[N-1]= q_{N-1}

        return self.compute_uniform_tangent(q_im1, quats, q_ip1)

    def squad(self, q0, a, b, q1, t):
        """
        Shoemake's "squad" interpolation for quaternion splines:
            squad(q0, a, b, q1; t) = slerp( slerp(q0, q1; t),
                                            slerp(a,   b;   t),
                                            2t(1-t) )
        where a, b are tangential control quaternions for q0, q1.
        """
        s1 = self.q_slerp(q0, q1, t)
        s2 = self.q_slerp(a,   b,   t)
        alpha = 2.0*t*(1.0 - t)
        return self.q_slerp(s1, s2, alpha)
    
    def uniform_cr_spline(self, quats, num_samples_per_segment=10):
        """
        Given a list of keyframe quaternions quats (each a torch 1D tensor [4]),
        compute a "Uniform Catmull–Rom–like" quaternion spline through them.

        Returns:
          A list (Python list) of interpolated quaternions (torch tensors),
          including all segment endpoints.

        Each interior qi gets a tangent T_i using neighbors q_{i-1}, q_i, q_{i+1}.
        For boundary tangents, we replicate the end quaternions.
        """
        n = quats.shape[0]
        if n < 2:
            return quats.unsqueeze(0)  # not enough quats to interpolate

        # Precompute tangents
        tangents = self.compute_all_uniform_tangents(quats)

        # Interpolate each segment [qi, q_{i+1}]
        q0 = quats[:-1].unsqueeze(1) 
        q1 = quats[1:].unsqueeze(1)  
        a = tangents[:-1].unsqueeze(1) 
        b = tangents[1:].unsqueeze(1)  

        t_vals = torch.linspace(0.0, 1.0, num_samples_per_segment, device=quats.device, dtype=quats.dtype)
        t_vals = t_vals.view(1, -1, 1)

        out = self.squad(q0, a, b, q1, t_vals)
        return out


    def forward(self, pred, targ, cond=None, scene_id=None, norm_params=None):
        loss, err_t, err_smooth, err_geo, err_r, err_dtw, err_hausdorff, err_frechet, err_chamfer = self._loss(pred, targ, cond, scene_id, norm_params)
        
        # Uncomment if you want to use other losses (also uncomment within _loss)
        info = {
            'trans. error': err_t.item(),
            'smoothness error': err_smooth.item(),
            # 'dtw': err_dtw.item(),
            # 'hausdorff': err_hausdorff.item(),
            # 'frechet': err_frechet.item(),
            # 'chamfer': err_chamfer.item(),
            'quat. dist.': err_r.item(),
            'geodesic dist.': err_geo.item(),
        }
        
        return loss, info

    def _loss(self, pred, targ, cond=None, scene_id=None, norm_params=None):
        def poly_eval(coeffs, x):
            """
            Evaluates a polynomial (with highest-degree term first) at points x.
            coeffs: 2D tensor of shape [num_polynomials, degree + 1], highest-degree term first.
            x: 1D tensor of points at which to evaluate the polynomial.
            Returns:
            2D tensor of shape [num_polynomials, len(x)], containing p(x).
            """
            x_powers = torch.stack([x**i for i in range(coeffs.shape[-1] - 1, -1, -1)], dim=-1)
            x_powers = x_powers.to(torch.float64).to(coeffs.device)
            y = torch.matmul(coeffs, x_powers.T)
            return y
        
        # Make sure the dtype is float64
        pred = pred.to(torch.float64)
        targ = targ.to(torch.float64)

        # Rescale the translations
        if scene_id is not None and norm_params is not None:
            scene_id = scene_id.item()
            scene_scale = self.scales[str(scene_id)]
            scene_scale = norm_params['scale'][0] * scene_scale
            pred[..., :3] = pred[..., :3] * scene_scale
            targ[..., :3] = targ[..., :3] * scene_scale
            # print(pred[..., :3].max(), targ[..., :3].max())

        # We only consider interpolated points for loss calculation
        candidate_idxs = sorted(cond.keys())
        pred = pred[:, candidate_idxs[0] : candidate_idxs[-1] + 1, :]
        targ = targ[:, candidate_idxs[0] : candidate_idxs[-1] + 1, :]
        
        pred_trans = pred[..., :3]
        pred_quat = pred[..., 3:7]
        targ_trans = targ[..., :3]
        targ_quat = targ[..., 3:7]

        pred_coeffs = self.compute_spline_coeffs(pred_trans)
        targ_coeffs = self.compute_spline_coeffs(targ_trans)
        
        n_points = 2000

        # Distribute sample points among intervals
        dists = torch.norm(targ_trans[:, 1:, :] - targ_trans[:, :-1, :], dim=-1).reshape(-1)
        dists_c = torch.zeros(len(candidate_idxs) - 1, device=pred.device)
        for i in range(len(candidate_idxs) - 1):
            dists_c[i] = dists[candidate_idxs[i]:candidate_idxs[i+1]].sum()
        
        weights_c = dists_c / dists_c.sum()
        scaled_c = weights_c * n_points
        points_c = torch.floor(scaled_c).int()

        while points_c.sum() < n_points:
            idx = torch.argmax(scaled_c - points_c)
            points_c[idx] += 1 
        
        # Calculate the spline loss
        sample_points = 50
        x = torch.linspace(0, 1, sample_points, device=pred.device)
        pred_spline = poly_eval(pred_coeffs, x).permute(0, 1, 3, 2).reshape(-1, sample_points, 3)
        targ_spline = poly_eval(targ_coeffs, x).permute(0, 1, 3, 2).reshape(-1, sample_points, 3)

        indexes = []
        start_idx = candidate_idxs[0]
        for c, (idx_i0, idx_i1) in enumerate(zip(candidate_idxs[:-1], candidate_idxs[1:])):
            p = points_c[c]
            total_dist = dists_c[c]
            dist_arr = dists[idx_i0 - start_idx : idx_i1 - start_idx]
            
            step_distances = (dist_arr / sample_points).repeat_interleave(sample_points)
            cumul_distances = step_distances.cumsum(dim=0)

            dist_per_pick = total_dist / p
            pick_targets = torch.arange(1, p + 1, device=dists.device) * dist_per_pick

            pick_idxs = torch.searchsorted(cumul_distances, pick_targets, right=True)
            pick_idxs = torch.clamp(pick_idxs, max=len(cumul_distances) - 1)


            indexes_1d = torch.zeros_like(step_distances)
            indexes_1d[pick_idxs] = 1

            indexes_2d = indexes_1d.view(len(dist_arr), sample_points)

            indexes.append(indexes_2d)

        indexes = torch.cat(indexes)[1: -1] # The first and last candidates don't have spline representations
     
        indexes_trans = torch.stack([indexes for _ in range(3)], dim=-1)
        indexes_quat = torch.stack([indexes for _ in range(4)], dim=-1)

        indexes_trans = indexes_trans.to(torch.bool)
        indexes_quat = indexes_quat.to(torch.bool)

        pred_trans_selected_values = pred_spline[indexes_trans]
        targ_trans_selected_values = targ_spline[indexes_trans]

        pred_trans_selected_values = pred_trans_selected_values.reshape(-1, 3)
        targ_trans_selected_values = targ_trans_selected_values.reshape(-1, 3)

        # Calculate the loss for quaternions
        pred_quat = pred_quat / pred_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        targ_quat = targ_quat / targ_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        targ_quat_spline = self.uniform_cr_spline(targ_quat.reshape(-1, 4), num_samples_per_segment=sample_points)
        pred_quat_spline = self.uniform_cr_spline(pred_quat.reshape(-1, 4), num_samples_per_segment=sample_points)

        
        targ_quat_spline = targ_quat_spline[1:-1]
        pred_quat_spline = pred_quat_spline[1:-1]


        pred_quat_selected_values = pred_quat_spline[indexes_quat]
        targ_quat_selected_values = targ_quat_spline[indexes_quat]

        pred_quat_selected_values = pred_quat_selected_values.reshape(-1, 4)
        targ_quat_selected_values = targ_quat_selected_values.reshape(-1, 4)

        # Calculate the geodesic loss
        pred_rot = quaternion_to_matrix(pred_quat_selected_values).reshape(-1, 3, 3)
        targ_rot = quaternion_to_matrix(targ_quat_selected_values).reshape(-1, 3, 3)

        eps = 1e-12
        r2r1 = pred_rot @ targ_rot.permute(0, 2, 1)
        trace = torch.diagonal(r2r1, dim1=-2, dim2=-1).sum(-1)
        trace = torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps)
        geodesic_loss = torch.acos(trace).mean()

        # Calculate the rotation error
        dot_product = torch.sum(pred_quat_selected_values * targ_quat_selected_values, dim=-1).clamp(min=-1.0 + eps, max=1.0 - eps)
        quaternion_dist = 1 - (dot_product ** 2).mean()

        # Calculate the L2 loss
        l2_loss = F.mse_loss(pred_trans_selected_values, targ_trans_selected_values, reduction='mean')

        # Calculate the smoothness loss for translation and quaternion
        smoothness_multiplier = 10 ** 2 # Empirically determined multiplier for smoothness loss
        weight_acceleration = 0.1 
        weight_jerk = 0.05 

        pos_acc = pred_trans_selected_values[2:, :] - 2 * pred_trans_selected_values[1:-1, :] + pred_trans_selected_values[:-2, :]
        pos_jerk = pred_trans_selected_values[3:, :] - 3 * pred_trans_selected_values[2:-1, :] + 3 * pred_trans_selected_values[1:-2, :] - pred_trans_selected_values[:-3, :]

        pos_acceleration_loss = torch.mean(pos_acc ** 2)
        pos_jerk_loss = torch.mean(pos_jerk ** 2)

        q0 = pred_quat_selected_values[:-1, :]
        q1 = pred_quat_selected_values[1:, :]
        sign = torch.where((q0 * q1).sum(dim=-1) < 0, -1.0, 1.0)
        q1 = sign.unsqueeze(-1) * q1

        dq = self.q_multiply(q1, self.q_inverse(q0))
        theta = 2 * torch.acos(torch.clamp(dq[..., 0], -1.0 + 1e-8, 1.0 - 1e-8))

        rot_acc  = theta[2:] - 2*theta[1:-1] + theta[:-2]
        rot_jerk = theta[3:] - 3*theta[2:-1] + 3*theta[1:-2] - theta[:-3] 

        rot_acceleration_loss = torch.mean(rot_acc ** 2)
        rot_jerk_loss = torch.mean(rot_jerk ** 2)

        alpha_rot = 0.1 # <-- can be tuned (e.g. 0.1 … 10)

        acceleration_loss = pos_acceleration_loss + alpha_rot * rot_acceleration_loss
        jerk_loss         = pos_jerk_loss + alpha_rot * rot_jerk_loss

        smoothness_loss = (
            weight_acceleration * acceleration_loss
          + weight_jerk        * jerk_loss
        ) * smoothness_multiplier


        # Calculate the spline loss
        l2_multiplier = 10.0 # Empirically determined multiplier for L2 loss
        l2_multiplier = 1.0 if norm_params is not None else l2_multiplier # If norm_params is provided, we assume that we are testing the model and do not want to scale the loss

        spline_loss = l2_multiplier * (l2_loss + smoothness_loss) + geodesic_loss + quaternion_dist

        dtw_loss, hausdorff_loss, frechet_loss, chamfer_loss = None, None, None, None

        # Uncomment these lines if you want to use other losses
        '''
        dtw = DynamicTimeWarpingLoss()
        dtw_loss, _ = dtw.forward(pred_trans_selected_values.reshape(-1, 3), targ_trans_selected_values.reshape(-1, 3))

        hausdorff = HausdorffDistanceLoss()
        hausdorff_loss, _ = hausdorff.forward(pred_trans_selected_values.reshape(-1, 3), targ_trans_selected_values.reshape(-1, 3))

        frec = FrechetDistanceLoss()
        frechet_loss, _ = frec.forward(pred_trans_selected_values.reshape(-1, 3), targ_trans_selected_values.reshape(-1, 3))

        chamfer = ChamferDistanceLoss()
        chamfer_loss, _ = chamfer.forward(pred_trans_selected_values.reshape(-1, 3), targ_trans_selected_values.reshape(-1, 3))
        '''

        return spline_loss, l2_loss, smoothness_loss, geodesic_loss, quaternion_dist, dtw_loss, hausdorff_loss, frechet_loss, chamfer_loss


class DynamicTimeWarpingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _dtw_distance(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """
        Computes the DTW distance between two 2D tensors (T x D), 
        where T is sequence length and D is feature dimension.
        """
        # seq1, seq2 shapes: (time_steps, feature_dim)
        n, m = seq1.size(0), seq2.size(0)

        # Cost matrix (pairwise distances between all elements)
        cost = torch.zeros(n, m, device=seq1.device, dtype=seq1.dtype)
        for i in range(n):
            for j in range(m):
                cost[i, j] = torch.norm(seq1[i] - seq2[j], p=2)

        # Accumulated cost matrix
        dist = torch.full((n + 1, m + 1), float('inf'), 
                          device=seq1.device, dtype=seq1.dtype)
        dist[0, 0] = 0.0

        # Populate the DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist[i, j] = cost[i - 1, j - 1] + torch.min(
                    torch.min(
                    dist[i - 1, j],   # Insertion
                    dist[i, j - 1],   # Deletion
                    ),
                    dist[i - 1, j - 1]# Match
                )

        return dist[n, m]

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """
        Compute the average DTW loss over a batch of sequences.
        
        pred, targ shapes: (batch_size, T, D)
        """
        # Ensure shapes match in batch dimension
        assert pred.size(0) == targ.size(0), "Batch sizes must match."

        # Compute DTW distance per sample in the batch
        distances = []
        for b in range(pred.size(0)):
            seq1 = pred[b]
            seq2 = targ[b]
            dtw_val = self._dtw_distance(seq1, seq2)
            distances.append(dtw_val)

        # Stack and take mean to get scalar loss
        dtw_loss = torch.stack(distances).mean()
        return dtw_loss

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        """
        Returns a tuple: (loss, info_dict), 
        where loss is a scalar tensor and info_dict is a dictionary 
        of extra information (e.g., loss components).
        """
        loss = self._loss(pred, targ)

        info = {
            'dtw': loss.item()
        }

        return loss, info

class HausdorffDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _hausdorff_distance(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hausdorff distance between two 2D tensors (N x D),
        where N is the number of points and D is the feature dimension.

        The Hausdorff distance H(A,B) between two sets A and B is defined as:
            H(A, B) = max( h(A, B), h(B, A) ),
        where
            h(A, B) = max_{a in A} min_{b in B} d(a, b).

        Here, d(a, b) is the Euclidean distance between points a and b.
        """
        n, m = set1.size(0), set2.size(0)

        # Compute pairwise distances
        cost = torch.zeros(n, m, device=set1.device, dtype=set1.dtype)
        for i in range(n):
            for j in range(m):
                cost[i, j] = torch.norm(set1[i] - set2[j], p=2)

        # Forward direction: for each point in set1, find distance to closest point in set2
        forward_min = cost.min(dim=1)[0] 
        forward_hausdorff = forward_min.max() 

        # Backward direction: for each point in set2, find distance to closest point in set1
        backward_min = cost.min(dim=0)[0] 
        backward_hausdorff = backward_min.max()

        # Hausdorff distance is the max of the two
        hausdorff_dist = torch.max(forward_hausdorff, backward_hausdorff)
        return hausdorff_dist

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """
        Compute the average Hausdorff distance over a batch of point sets.

        pred, targ shapes: (batch_size, N, D)
        """
        # Ensure shapes match in batch dimension
        assert pred.size(0) == targ.size(0), "Batch sizes must match."

        distances = []
        for b in range(pred.size(0)):
            set1 = pred[b]
            set2 = targ[b]
            h_dist = self._hausdorff_distance(set1, set2)
            distances.append(h_dist)

        # Stack and take mean to get scalar loss
        hausdorff_loss = torch.stack(distances).mean()
        return hausdorff_loss

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        """
        Returns a tuple: (loss, info_dict), 
        where loss is a scalar tensor and info_dict is a dictionary 
        of extra information (e.g., distance components).
        """
        loss = self._loss(pred, targ)

        info = {
            'hausdorff': loss.item()
        }

        return loss, info

class FrechetDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _frechet_distance(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """
        Computes the (discrete) Fréchet distance between two 2D tensors (T x D),
        where T is the sequence length and D is the feature dimension.

        The Fréchet distance between two curves in discrete form can be computed
        by filling in a DP table “ca” where:

            ca[i, j] = max( d(seq1[i], seq2[j]),
                            min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]) )

        with boundary conditions handled appropriately.
        Here, d(seq1[i], seq2[j]) is the Euclidean distance.
        """
        n, m = seq1.size(0), seq2.size(0)

        # Cost matrix (pairwise distances between all elements)
        cost = torch.zeros(n, m, device=seq1.device, dtype=seq1.dtype)
        for i in range(n):
            for j in range(m):
                cost[i, j] = torch.norm(seq1[i] - seq2[j], p=2)

        # DP matrix for the Fréchet distance
        ca = torch.full((n, m), float('inf'), device=seq1.device, dtype=seq1.dtype)
        ca[0, 0] = cost[0, 0]

        # Initialize first row
        for i in range(1, n):
            ca[i, 0] = torch.max(ca[i - 1, 0], cost[i, 0])

        # Initialize first column
        for j in range(1, m):
            ca[0, j] = torch.max(ca[0, j - 1], cost[0, j])

        # Populate the DP table
        for i in range(1, n):
            for j in range(1, m):
                ca[i, j] = torch.max(
                    cost[i, j],
                    torch.min(
                        torch.min(
                        ca[i - 1, j],
                        ca[i, j - 1],
                        ),
                        ca[i - 1, j - 1]
                    )
                )

        return ca[n - 1, m - 1]

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """
        Compute the average Fréchet distance over a batch of sequences.

        pred, targ shapes: (batch_size, T, D)
        """
        # Ensure shapes match in batch dimension
        assert pred.size(0) == targ.size(0), "Batch sizes must match."

        distances = []
        for b in range(pred.size(0)):
            seq1 = pred[b]
            seq2 = targ[b]
            fd_val = self._frechet_distance(seq1, seq2)
            distances.append(fd_val)

        # Stack and take mean to get scalar loss
        frechet_loss = torch.stack(distances).mean()
        return frechet_loss

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        """
        Returns a tuple: (loss, info_dict),
        where loss is a scalar tensor and info_dict is a dictionary
        of extra information (e.g., distance components).
        """
        loss = self._loss(pred, targ)
        info = {
            'frechet': loss.item()
        }
        return loss, info

class ChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _chamfer_distance(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetrical Chamfer distance between
        two 2D tensors (N x D), where N is the number of points
        and D is the feature dimension.

        The Chamfer distance between two point sets A and B is often defined as:

            d_chamfer(A, B) = 1/|A| ∑_{a ∈ A} min_{b ∈ B} ‖a - b‖₂
                              + 1/|B| ∑_{b ∈ B} min_{a ∈ A} ‖b - a‖₂,

        where ‖·‖₂ is the Euclidean distance.
        """
        # set1, set2 shapes: (num_points, feature_dim)
        n, m = set1.size(0), set2.size(0)

        cost = torch.zeros(n, m, device=set1.device, dtype=set1.dtype)
        for i in range(n):
            for j in range(m):
                cost[i, j] = torch.norm(set1[i] - set2[j], p=2)

        # For each point in set1, find distance to the closest point in set2
        forward_min = cost.min(dim=1)[0]   # shape: (n,)
        forward_mean = forward_min.mean()

        # For each point in set2, find distance to the closest point in set1
        backward_min = cost.min(dim=0)[0]  # shape: (m,)
        backward_mean = backward_min.mean()

        chamfer_dist = forward_mean + backward_mean
        return chamfer_dist

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """
        Compute the average Chamfer distance over a batch of point sets.

        pred, targ shapes: (batch_size, N, D)
        """
        # Ensure shapes match in batch dimension
        assert pred.size(0) == targ.size(0), "Batch sizes must match."

        distances = []
        for b in range(pred.size(0)):
            set1 = pred[b]
            set2 = targ[b]
            distance_val = self._chamfer_distance(set1, set2)
            distances.append(distance_val)

        # Combine into a single scalar
        chamfer_loss = torch.stack(distances).mean()
        return chamfer_loss

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        """
        Returns a tuple: (loss, info_dict),
        where 'loss' is a scalar tensor and 'info_dict' is a dictionary
        of extra information (e.g., distance components).
        """
        loss = self._loss(pred, targ)
        info = {
            'chamfer': loss.item()
        }
        return loss, info
        

def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)

    if dot < 0.0:
        q2 = -q2
        dot = -dot
    # If dot is very close to 1, use linear interpolation
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result = result / np.linalg.norm(result)
        return result
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    return q1 * np.cos(theta) + q3 * np.sin(theta)

def catmull_rom_spline_with_rotation(control_points, timepoints, horizon):
    """Compute Catmull-Rom spline for both position and quaternion rotation."""
    spline_points = []
    # Extrapolate the initial points
    if timepoints[0] != 0:
        for t in range(timepoints[0]):
            x = control_points[0][0]
            y = control_points[0][1]
            z = control_points[0][2]
            q = control_points[0][3:7]
            spline_points.append(np.concatenate([np.array([x, y, z]), q]))

    #Linear interpolate between 0th and 1th control points
    for t in np.linspace(0, 1, timepoints[1] - timepoints[0] + 1):
        x = control_points[0][0] + t * (control_points[1][0] - control_points[0][0])
        y = control_points[0][1] + t * (control_points[1][1] - control_points[0][1])
        z = control_points[0][2] + t * (control_points[1][2] - control_points[0][2])
        q = slerp(control_points[0][3:7], control_points[1][3:7], t)
        spline_points.append(np.concatenate([np.array([x, y, z]), q]))


    # Iterate over the control points
    for i in range(1, len(control_points) - 2):
        P0 = control_points[i-1][:3] 
        P1 = control_points[i][:3]    
        P2 = control_points[i+1][:3]  
        P3 = control_points[i+2][:3]  
        Q0 = control_points[i-1][3:7]  
        Q1 = control_points[i][3:7]    
        Q2 = control_points[i+1][3:7]  
        Q3 = control_points[i+2][3:7]
    
        # Interpolate position (using Catmull-Rom spline)
        for idx, t in enumerate(np.linspace(0, 1, timepoints[i+1] - timepoints[i] + 1)):
            if idx == 0:
                continue

            x = 0.5 * ((2 * P1[0]) + (-P0[0] + P2[0]) * t + 
                       (2 * P0[0] - 5 * P1[0] + 4 * P2[0] - P3[0]) * t**2 + 
                       (-P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0]) * t**3)
            y = 0.5 * ((2 * P1[1]) + (-P0[1] + P2[1]) * t + 
                       (2 * P0[1] - 5 * P1[1] + 4 * P2[1] - P3[1]) * t**2 + 
                       (-P0[1] + 3 * P1[1] - 3 * P2[1] + P3[1]) * t**3)
            z = 0.5 * ((2 * P1[2]) + (-P0[2] + P2[2]) * t + 
                       (2 * P0[2] - 5 * P1[2] + 4 * P2[2] - P3[2]) * t**2 + 
                       (-P0[2] + 3 * P1[2] - 3 * P2[2] + P3[2]) * t**3)
            q = slerp(Q1, Q2, t)
            spline_points.append(np.concatenate([np.array([x, y, z]), q]))
        
    #Linear interpolate between 2nd last and last control points
    for idx, t in enumerate(np.linspace(0, 1, timepoints[-1] - timepoints[-2] + 1)):
        if idx == 0:
            continue
        x = control_points[-2][0] + t * (control_points[-1][0] - control_points[-2][0])
        y = control_points[-2][1] + t * (control_points[-1][1] - control_points[-2][1])
        z = control_points[-2][2] + t * (control_points[-1][2] - control_points[-2][2])
        q = slerp(control_points[-2][3:7], control_points[-1][3:7], t)
        spline_points.append(np.concatenate([np.array([x, y, z]), q]))

    # Extrapolate the rest of the points
    if timepoints[-1] != horizon:
        for t in range(timepoints[-1] + 1, horizon):
            x = control_points[-1][0]
            y = control_points[-1][1]
            z = control_points[-1][2]
            q = control_points[-1][3:7]
            spline_points.append(np.concatenate([np.array([x, y, z]), q]))
    
    stacked_spline_points = np.stack(spline_points, axis=0)

    if control_points.shape[1] != 7:
        stacked_spline_points = np.concatenate([stacked_spline_points, np.zeros((stacked_spline_points.shape[0], 1))], axis=1)
    
    
    return stacked_spline_points

def catmull_rom_loss(trajectories, conditions, loss_fc):
    '''
        loss for catmull-rom interpolation
    '''        
    batch_size, horizon, transition = trajectories.shape
    
    # Extract known indices and values
    known_indices = np.array(list(conditions.keys()), dtype=int)
    
    # candidate_no x batch_size x dim
    known_values = np.stack([c.cpu().numpy() for c in conditions.values()], axis=0)
    known_values = np.moveaxis(known_values, 0, 1)
    
    # Sort the timepoints
    sorted_indices = np.argsort(known_indices)
    known_indices = known_indices[sorted_indices]
    known_values = known_values[:, sorted_indices]
    spline_points = np.array([catmull_rom_spline_with_rotation(known_values[b], known_indices, horizon) for b in range(batch_size)])
    
    # Convert to tensor and move to the same device as trajectories
    spline_points = torch.tensor(spline_points, dtype=torch.float64, device=trajectories.device)
    assert spline_points.shape == trajectories.shape, f"Shape mismatch: {spline_points.shape} != {trajectories.shape}"
    return loss_fc(spline_points, trajectories)

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'geodesic_l2': GeodesicL2Loss,
    'rotation_translation': RotationTranslationLoss,
    'spline': SplineLoss,
}
