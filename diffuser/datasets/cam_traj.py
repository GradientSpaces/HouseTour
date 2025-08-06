from collections import namedtuple
import numpy as np
import torch
import json
import random
from scipy.spatial.transform import Rotation as R

Batch = namedtuple('Batch', 'trajectories conditions scene_id scale')


class CameraTrajectoriesDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_traj):
        
        with open(path_to_traj, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)
    
    def normalize_and_align(self, trajectories, candidates_trajectory):
        """
        Normalizes the first 3 dimensions (x,y,z) of `trajectories` (2D) and `candidates_trajectory` (1D) to [-1, 1].
        """
        min_vals = np.min(trajectories[:, :3], axis=0)  # Min of first 3 dimensions
        max_vals = np.max(trajectories[:, :3], axis=0)  # Max of first 3 dimensions

        # Avoid division by zero in case of constant values
        scale = max_vals - min_vals
        scale[scale == 0] = 1

        def norm_2d(x):
            return np.hstack([
                2 * (x[:, :3] - min_vals) / scale - 1,
                x[:, 3:]
            ])

        def norm_1d(x):
            return np.concatenate([
                2 * (x[:3] - min_vals) / scale - 1,
                x[3:]
            ])
        
        trajectories = norm_2d(trajectories)
        candidates_trajectory = {idx: norm_1d(c) for idx, c in candidates_trajectory.items()}

        return trajectories, candidates_trajectory, {"min_vals" : min_vals, "scale" : scale}


    def __getitem__(self, idx):
        """
        Get a single trajectory with merged target and candidate points.
        In this version, the target and candidate trajectories (and their associated times
        and images) are merged immediately (with a binary label indicating each point's origin)
        and then candidate points are downsampled if necessary. Finally, the total number of
        points is adjusted (by removing target points with minimal consecutive difference)
        so that the merged sequence length is divisible by eight.
        """
        dp = self.data[idx]
        scene_id, trajectory = next(iter(dp.items()))
    
        # -------------------------------------------------------------------------
        # 1. Extract data for candidates and target.
        # -------------------------------------------------------------------------
        candidates = trajectory['candidates']
        target = trajectory['trajectory']

        candidates_tvec = [c['tvec'] for c in candidates]
        target_tvec = [t['tvec'] for t in target]
    
        candidates_qvec = [c['qvec'] / np.linalg.norm(c['qvec']) for c in candidates]
        target_qvec = [t['qvec'] / np.linalg.norm(t['qvec']) for t in target]
    
        candidates_time = [c['time'] for c in candidates]
        target_time = [t['time'] for t in target]
    
        candidates_tvec = np.array(candidates_tvec, dtype="float32")
        target_tvec = np.array(target_tvec, dtype="float32")
        candidates_qvec = np.array(candidates_qvec, dtype="float32")
        target_qvec = np.array(target_qvec, dtype="float32")
        candidates_time = np.array(candidates_time, dtype="float32")
        target_time = np.array(target_time, dtype="float32")
    
        # -------------------------------------------------------------------------
        # 2. Create pose arrays (translation + rotation) and add binary labels.
        #    Label 1 indicates a candidate point and 0 indicates a target point.
        # -------------------------------------------------------------------------
        candidates_pose = np.concatenate([candidates_tvec, candidates_qvec], axis=1)
        target_pose = np.concatenate([target_tvec, target_qvec], axis=1)
    
        candidates_pose = np.concatenate([candidates_pose,
                                          np.ones((len(candidates_pose), 1), dtype="float32")],
                                          axis=1)
        target_pose = np.concatenate([target_pose,
                                      np.zeros((len(target_pose), 1), dtype="float32")],
                                      axis=1)
    
        # -------------------------------------------------------------------------
        # 3. Merge target and candidate trajectories, times, and images.
        # -------------------------------------------------------------------------
        merged_pose = np.concatenate([target_pose, candidates_pose], axis=0)
        merged_time = np.concatenate([target_time, candidates_time], axis=0)
    
        # Sort the merged arrays by time.
        sort_idx = np.argsort(merged_time)
        merged_pose = merged_pose[sort_idx]
        merged_time = merged_time[sort_idx]
    
        # -------------------------------------------------------------------------
        # 4. Downsample candidate points (if necessary) from the merged trajectory.
        #    Only candidate points (binary label == 1) are considered for downsampling.
        # -------------------------------------------------------------------------
        candidate_mask = merged_pose[:, -1] == 1
        candidate_indices = np.where(candidate_mask)[0]
        candidate_count = len(candidate_indices)
    
        # Determine the desired number of candidate points using a divisor-based logic (this can be tuned based on required observation sparsity).
        divisor = 10
        total_points = len(merged_pose)
        n_with_interval = total_points // divisor
        N = np.random.randint(min(n_with_interval, candidate_count), candidate_count + 1)
    
        if candidate_count > N:
            # Select evenly spaced candidate indices among the candidate indices.
            selected_candidate_indices = candidate_indices[np.linspace(0, candidate_count - 1, N, dtype=int)]
            # Build a mask that keeps all target points (label 0) and only the selected candidate points.
            keep_mask = np.ones(len(merged_pose), dtype=bool)
            for idx in candidate_indices:
                if idx not in selected_candidate_indices:
                    keep_mask[idx] = False
            merged_pose = merged_pose[keep_mask]
            merged_time = merged_time[keep_mask]
    
        # -------------------------------------------------------------------------
        # 5. Ensure the merged trajectory's length is divisible by eight (this is necessary for the downsampling step inside Residual Diffuser).
        #    Remove target points (label == 0) having the smallest consecutive difference.
        # -------------------------------------------------------------------------
        while len(merged_pose) % 8 != 0:
            target_indices = np.where(merged_pose[:, -1] == 0)[0]
            if len(target_indices) == 0:
                break
            # Compute consecutive differences for the pose components (ignoring the binary label).
            diffs = np.linalg.norm(np.diff(merged_pose[:, :7], axis=0), axis=1)
            # Consider removal only for target points that are not at the very start.
            valid_target_indices = [i for i in target_indices if i > 0]
            if valid_target_indices:
                # For each valid target point, associate it with the distance between it and its predecessor.
                candidate_diffs = {i: diffs[i - 1] for i in valid_target_indices}
                remove_idx = min(candidate_diffs, key=candidate_diffs.get)
            else:
                remove_idx = target_indices[0]
            merged_pose = np.delete(merged_pose, remove_idx, axis=0)
            merged_time = np.delete(merged_time, remove_idx, axis=0)
        
        for h in range(1, len(merged_pose)):
            if np.dot(merged_pose[h-1, 3:7], merged_pose[h, 3:7]) < 0:
                merged_pose[h, 3:7] = -merged_pose[h, 3:7]
    
        # -------------------------------------------------------------------------
        # 6. Reconstruct the candidate dictionary from the merged pose.
        #    (Indices with label == 1 correspond to candidate points.)
        # -------------------------------------------------------------------------
        final_candidate_indices = np.where(merged_pose[:, -1] == 1)[0]
        candidates_dict = {idx: merged_pose[idx] for idx in final_candidate_indices}
    
        # -------------------------------------------------------------------------
        # 7. Normalize and align the merged trajectory and prepare the batch.
        # -------------------------------------------------------------------------
        # Calculate the dot product between consecutive quaternions, flip the sign if necessary
        
        trajectories, candidates_aligned, norm_dict = self.normalize_and_align(merged_pose, candidates_dict)
    
        batch = Batch(
            trajectories=trajectories,
            conditions=candidates_aligned,
            scene_id=np.array(int(scene_id)),
            scale=norm_dict,
        )
        return batch