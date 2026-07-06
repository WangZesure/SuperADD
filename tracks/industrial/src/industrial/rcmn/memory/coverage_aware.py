"""Coverage-aware memory construction (CMC).

Unlike kNN dedup (which only removes near-duplicates), CMC actively ensures
the memory bank covers the full diversity of normal patterns, including
rare but legitimate variations (lighting, position, texture).

Two-stage approach for practical feasibility:
  1. Rarity scoring: compute kNN density for all features (chunked, GPU)
  2. Rarity-weighted random sampling: sample target_size features with
     probability proportional to rarity (sparse regions over-sampled)

This replaces the original O(N × target_size) greedy farthest-point selection
(which was infeasible for N=300K, target_size=100K) with O(N) weighted
sampling — same asymptotic cost as random subsampling, but coverage-aware.

Optional diversity refinement: after weighted sampling, run one pass of
near-duplicate removal to avoid clustering in high-density regions.
"""

import numpy as np
import torch


def subsample(
    features: np.ndarray,
    target_size: int,
    device: str = 'cuda',
    **kwargs,
) -> np.ndarray:
    """Coverage-aware subsampling of normal feature vectors.

    Args:
        features:     (N, D) float32 feature vectors
        target_size:  desired memory bank size M
        device:       torch device
        **kwargs:
            knn_neighbors (int): k for density estimation, default 50
            query_batch (int): query batch for kNN, default 2048
            key_chunk (int): key chunk for distance, default 8192
            dedup (bool): near-duplicate removal after sampling, default True
            dedup_k (int): kNN k for dedup, default 5

    Returns:
        (M, D) float32 subsampled features
    """
    knn_k = kwargs.get('knn_neighbors', 50)
    query_batch = kwargs.get('query_batch', 2048)
    key_chunk = kwargs.get('key_chunk', 8192)
    do_dedup = kwargs.get('dedup', True)
    dedup_k = kwargs.get('dedup_k', 5)

    N = len(features)
    if N <= target_size:
        return features

    feats_gpu = torch.from_numpy(features).to(device)

    # --- Step 1: Rarity via kNN density ---
    rarity = _compute_knn_density_chunked(
        feats_gpu, knn_k, query_batch, key_chunk, device
    )
    # Normalize rarity to [0, 1]
    rarity = rarity / (rarity.max() + 1e-8)

    # --- Step 2: Rarity-weighted random sampling ---
    # Probability proportional to rarity (sparse regions over-sampled)
    # Add small epsilon to ensure all points have nonzero probability
    probs = rarity + 1e-6
    probs = probs / probs.sum()

    sampled_indices = torch.multinomial(
        probs, target_size, replacement=False
    ).cpu().numpy()

    selected = features[sampled_indices]

    # --- Step 3 (optional): Near-duplicate removal ---
    # Remove samples that are too close to each other, then top up
    if do_dedup and len(selected) > target_size * 0.9:
        selected = _dedup_refinement(
            selected, features, target_size, device, dedup_k, key_chunk
        )

    return selected


def _compute_knn_density_chunked(
    feats: torch.Tensor,
    k: int,
    query_batch: int,
    key_chunk: int,
    device: str,
) -> torch.Tensor:
    """Compute mean kNN distance for each feature (rarity proxy).

    Uses double chunking (query + key) to avoid OOM on large feature sets.
    """
    N, D = feats.shape
    densities = torch.zeros(N, device=device)

    for q_start in range(0, N, query_batch):
        q_end = min(q_start + query_batch, N)
        query = feats[q_start:q_end]

        topk_dists = torch.full((q_end - q_start, k + 1), float('inf'), device=device)

        for k_start in range(0, N, key_chunk):
            k_end = min(k_start + key_chunk, N)
            key = feats[k_start:k_end]
            dists = torch.cdist(query, key, compute_mode='use_mm_for_euclid_dist')
            merged = torch.cat([topk_dists, dists], dim=1)
            topk_dists, _ = torch.topk(merged, k=k + 1, dim=1, largest=False)

        topk_dists, _ = torch.sort(topk_dists, dim=1)
        mean_dist = topk_dists[:, 1:].mean(dim=-1)
        densities[q_start:q_end] = mean_dist

    return densities


def _dedup_refinement(
    selected: np.ndarray,
    all_features: np.ndarray,
    target_size: int,
    device: str,
    k: int,
    key_chunk: int,
) -> np.ndarray:
    """Remove near-duplicate samples, then top up with random draws.

    Lighter than full greedy FPS — just one pass of kNN-based dedup.
    """
    if len(selected) <= target_size:
        return selected

    sel_gpu = torch.from_numpy(selected).to(device)

    # Compute pairwise kNN distances within selected set
    N = len(sel_gpu)
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)

    for i in range(0, N, key_chunk):
        end = min(i + key_chunk, N)
        query = sel_gpu[i:end]
        dists, _ = torch.topk(
            torch.cdist(query, sel_gpu, compute_mode='use_mm_for_euclid_dist'),
            k=k + 1, dim=1, largest=False,
        )
        # If nearest neighbor (excluding self) is very close, mark for removal
        nn_dist = dists[:, 1]
        too_close = nn_dist < 1e-4
        # Don't remove if already removed
        for j in range(len(too_close)):
            if too_close[j] and keep_mask[i + j]:
                keep_mask[i + j] = False
                break  # Only remove one of the pair

    kept = selected[keep_mask.cpu().numpy()]

    # Top up if we removed too many
    if len(kept) < target_size:
        deficit = target_size - len(kept)
        remaining = np.setdiff1d(
            np.arange(len(all_features)),
            np.where(np.isin(all_features[:, 0], kept[:, 0]))[0],
            assume_unique=False,
        )
        if len(remaining) > 0:
            topup = np.random.choice(remaining, size=min(deficit, len(remaining)), replace=False)
            kept = np.vstack([kept, all_features[topup]])

    return kept[:target_size] if len(kept) >= target_size else kept