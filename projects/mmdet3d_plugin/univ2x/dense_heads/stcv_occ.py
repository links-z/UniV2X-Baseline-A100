"""
STCV-Occ: Selective Temporal Cooperative V2X Occupancy Prediction
Online Integration Module for UniV2X

This module implements patch-level cooperation utility prediction and
selective fusion, strictly following G3/G4 offline validation methodology.

Author: G5-0B Implementation
Date: 2026-10-02
"""

import torch
import torch.nn as nn
from typing import Tuple, List


class NeighborhoodMLP(nn.Module):
    """
    Cooperation Utility Predictor

    Architecture (strictly matching G3):
        Input: 127-dim feature
        Hidden: 256 → 128 → 64
        Output: 1-dim logit (sigmoid applied externally)

    Total parameters: 73,985
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(127, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (N, 127) or (127,) feature tensor

        Returns:
            logits: (N,) or scalar, NO sigmoid applied
        """
        was_single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            was_single = True

        logits = self.net(x).squeeze(-1)  # (N,) or (1,)

        # Return scalar for single input
        if was_single:
            logits = logits.squeeze()

        return logits


class STCVFeatureExtractor:
    """
    Extract 127-dim features for cooperation utility prediction

    Feature composition (strictly matching G3):
        - Local soft evidence: 80-dim (5 channels × 4×4 patch)
        - Horizon encoding: 5-dim (one-hot)
        - Spatial position: 2-dim (normalized patch center)
        - Neighborhood context: 40-dim (8 neighbors × 5 stats)
    """

    def __init__(self):
        pass

    def build_full_feature(self, pv_h, pi_h, warp):
        """
        Build 5-channel full feature map for a single horizon

        Args:
            pv_h: (200, 200) ego soft occupancy probability
            pi_h: (200, 200) aligned infra soft occupancy probability
            warp: (200, 200) warp valid mask

        Returns:
            full_feat: (5, 200, 200)
                [Pv, Pi, Pi-Pv, |Pi-Pv|, Mwarp]
        """
        full_feat = torch.stack([
            pv_h,
            pi_h,
            pi_h - pv_h,
            torch.abs(pi_h - pv_h),
            warp.to(dtype=pv_h.dtype),
        ], dim=0)

        return full_feat  # (5, 200, 200)

    def build_patch_mean_grid(self, full_feat):
        """
        Compute mean of each 4×4 patch for all 5 channels

        Args:
            full_feat: (5, 200, 200)

        Returns:
            mean_grid: (5, 50, 50)
        """
        # Reshape: (5, 200, 200) → (5, 50, 4, 50, 4)
        # Then mean over (2, 4) → (5, 50, 50)
        mean_grid = full_feat.reshape(5, 50, 4, 50, 4).mean(dim=(2, 4))
        return mean_grid

    def extract_local_80(self, full_feat, r, c):
        """
        Extract local soft evidence (80-dim)

        Args:
            full_feat: (5, 200, 200)
            r, c: patch row, col (0-49)

        Returns:
            local_feat: (80,) = flatten(5 × 4 × 4)
        """
        patch = full_feat[:, 4*r:4*r+4, 4*c:4*c+4]  # (5, 4, 4)
        return patch.flatten()  # (80,)

    def extract_horizon_5(self, h):
        """
        Extract horizon encoding (5-dim one-hot)

        Args:
            h: horizon index (0-4)

        Returns:
            horizon_feat: (5,)
        """
        horizon_feat = torch.zeros(5)
        horizon_feat[h] = 1.0
        return horizon_feat

    def extract_spatial_2(self, r, c):
        """
        Extract spatial position encoding (2-dim)

        Args:
            r, c: patch row, col (0-49)

        Returns:
            spatial_feat: (2,)
                u = 2*(c+0.5)/50 - 1  (col → u)
                v = 2*(r+0.5)/50 - 1  (row → v)
                Range: [-0.98, 0.98]
        """
        u = 2 * (c + 0.5) / 50 - 1
        v = 2 * (r + 0.5) / 50 - 1
        return torch.tensor([u, v], dtype=torch.float32)

    def extract_neighborhood_40(self, mean_grid, r, c, device, dtype):
        """
        Extract neighborhood context (40-dim)

        Args:
            mean_grid: (5, 50, 50) patch mean grid
            r, c: center patch row, col
            device, dtype: tensor properties

        Returns:
            neighbor_feat: (40,) = 8 neighbors × 5 stats
                Neighbors: NW, N, NE, W, E, SW, S, SE
                Stats: [mean(Pv), mean(Pi), mean(Pi-Pv), mean(|Pi-Pv|), mean(Mwarp)]
        """
        neighbors = []

        # 8 neighbors in fixed order
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < 50 and 0 <= nc < 50:
                vals = mean_grid[:, nr, nc]  # (5,)
            else:
                # Out of bounds: all zeros
                vals = torch.zeros(5, dtype=dtype, device=device)

            neighbors.append(vals)

        return torch.cat(neighbors)  # (40,)

    def extract_127_features(self, pv, pi, warp, h, r, c):
        """
        Extract complete 127-dim feature for a single patch

        Args:
            pv: (5, 200, 200) ego soft probability
            pi: (5, 200, 200) aligned infra soft probability
            warp: (200, 200) warp valid mask
            h: horizon index (0-4)
            r, c: patch row, col (0-49)

        Returns:
            feature: (127,)
                80: local soft evidence
                 5: horizon encoding
                 2: spatial position
                40: neighborhood context
        """
        device = pv.device
        dtype = pv.dtype

        # Build full feature for this horizon
        pv_h = pv[h]
        pi_h = pi[h]
        full_feat = self.build_full_feature(pv_h, pi_h, warp)

        # Build patch mean grid
        mean_grid = self.build_patch_mean_grid(full_feat)

        # Extract 4 components
        local_80 = self.extract_local_80(full_feat, r, c)  # (80,)
        horizon_5 = self.extract_horizon_5(h).to(device=device, dtype=dtype)  # (5,)
        spatial_2 = self.extract_spatial_2(r, c).to(device=device, dtype=dtype)  # (2,)
        neighbor_40 = self.extract_neighborhood_40(mean_grid, r, c, device, dtype)  # (40,)

        # Concatenate: 80 + 5 + 2 + 40 = 127
        feature = torch.cat([
            local_80,
            horizon_5,
            spatial_2,
            neighbor_40
        ])

        return feature  # (127,)


class STCVOcc(nn.Module):
    """
    STCV-Occ: Selective Temporal Cooperative V2X Occupancy Fusion

    Main module for online integration into UniV2X
    """

    def __init__(self, checkpoint_path, threshold=0.70):
        """
        Args:
            checkpoint_path: path to G3 trained predictor checkpoint
            threshold: cooperation utility threshold τ* (from G4)
        """
        super().__init__()

        self.predictor = NeighborhoodMLP()
        self.feature_extractor = STCVFeatureExtractor()
        self.threshold = threshold

        # Load G3 checkpoint
        self._load_checkpoint(checkpoint_path)

        print(f"[STCV-Occ] Initialized with τ={threshold}")

    def _load_checkpoint(self, checkpoint_path):
        """Load and verify G3 predictor checkpoint"""
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Load weights
        self.predictor.load_state_dict(ckpt['model'], strict=True)
        self.predictor.eval()

        # Verify parameter count
        n_params = sum(p.numel() for p in self.predictor.parameters())
        assert n_params == 73985, f"Expected 73985 params, got {n_params}"

        print(f"[STCV-Occ] Loaded G3 checkpoint: {checkpoint_path}")
        print(f"[STCV-Occ] Predictor parameters: {n_params}")

    def forward(self, pv, pi, ov, oi, warp):
        """
        Forward pass with batch support

        Args:
            pv: (B, 5, 200, 200) ego soft occupancy probability
            pi: (B, 5, 200, 200) aligned infra soft occupancy probability
            ov: (B, 5, 200, 200) ego binary occupancy
            oi: (B, 5, 200, 200) infra binary occupancy
            warp: (B, 200, 200) warp valid mask (no horizon dim)

        Returns:
            fused_occ: (B, 5, 200, 200) learned selective fusion result
        """
        B = pv.shape[0]
        outputs = []

        for b in range(B):
            out_b = self._forward_single(
                pv[b],    # (5, 200, 200)
                pi[b],    # (5, 200, 200)
                ov[b],    # (5, 200, 200)
                oi[b],    # (5, 200, 200)
                warp[b],  # (200, 200)
            )
            outputs.append(out_b)

        return torch.stack(outputs, dim=0)  # (B, 5, 200, 200)

    def _forward_single(self, pv, pi, ov, oi, warp):
        """
        Process single sample

        Args:
            pv, pi: (5, 200, 200)
            ov, oi: (5, 200, 200)
            warp: (200, 200)

        Returns:
            fused_occ: (5, 200, 200)
        """
        # Step 1: Extract candidate patches and features
        candidate_indices, features = self.extract_candidates_and_features(
            pv, pi, ov, oi, warp
        )

        # Step 2: Predict cooperation utility
        if len(candidate_indices) > 0:
            with torch.no_grad():
                logits = self.predictor(features)  # (N,)
                scores = torch.sigmoid(logits)      # (N,)
                accept = (scores > self.threshold)  # (N,)
        else:
            accept = torch.empty(0, dtype=torch.bool, device=pv.device)

        # Step 3: Selective fusion
        fused_occ = self.selective_fusion(
            ov, oi, candidate_indices, accept
        )

        return fused_occ

    def extract_candidates_and_features(self, pv, pi, ov, oi, warp):
        """
        Extract candidate patches and their features

        Candidate definition: ADD_R = (¬Ov ∧ Oi ∧ Mwarp) in patch
        Traversal order: h → r → c (fixed, matching G3)

        Args:
            pv, pi: (5, 200, 200) soft probability
            ov, oi: (5, 200, 200) binary occupancy
            warp: (200, 200) warp valid mask

        Returns:
            candidate_indices: List[(h, r, c)]
            features: (N_candidates, 127)
        """
        candidate_indices = []

        # Fixed order: horizon → row → col
        for h in range(5):
            for r in range(50):
                for c in range(50):
                    # Extract patch region
                    patch_ov = ov[h, 4*r:4*r+4, 4*c:4*c+4]
                    patch_oi = oi[h, 4*r:4*r+4, 4*c:4*c+4]
                    patch_warp = warp[4*r:4*r+4, 4*c:4*c+4]  # No horizon dim

                    # Candidate condition: ADD = (¬Ov ∧ Oi ∧ Mwarp)
                    add_patch = (~patch_ov.bool()) & patch_oi.bool() & patch_warp.bool()

                    if add_patch.sum() > 0:
                        candidate_indices.append((h, r, c))

        # Extract features for all candidates
        features = []
        for h, r, c in candidate_indices:
            feat = self.feature_extractor.extract_127_features(
                pv, pi, warp, h, r, c
            )
            features.append(feat)

        if len(features) > 0:
            features = torch.stack(features)  # (N, 127)
        else:
            features = torch.empty(0, 127, device=pv.device, dtype=pv.dtype)

        return candidate_indices, features

    def selective_fusion(self, ov, oi, candidate_indices, accept_mask):
        """
        Selective fusion: O_learned = Ov ∨ (aR ∧ Oi)

        Args:
            ov, oi: (5, 200, 200) binary occupancy
            candidate_indices: List[(h, r, c)]
            accept_mask: (N_candidates,) boolean

        Returns:
            fused_occ: (5, 200, 200)
        """
        # Start from ego occupancy
        fused_occ = ov.clone()

        # For each accepted candidate, OR with infra
        for idx, (h, r, c) in enumerate(candidate_indices):
            if accept_mask[idx]:
                # Patch region
                patch_ov = fused_occ[h, 4*r:4*r+4, 4*c:4*c+4]
                patch_oi = oi[h, 4*r:4*r+4, 4*c:4*c+4]

                # OR fusion
                fused_occ[h, 4*r:4*r+4, 4*c:4*c+4] = patch_ov | patch_oi

        return fused_occ


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'NeighborhoodMLP',
    'STCVFeatureExtractor',
    'STCVOcc',
]
