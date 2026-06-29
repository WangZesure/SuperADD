"""Reliability-aware Scale Fusion (RSF).

Instead of equal-weight averaging across scales/layers, RSF estimates per-pixel
reliability for each (scale, layer) map and fuses with softmax-weighted combination.

Reliability cues (all computed without anomaly labels):

1. Retrieval margin (R_margin):
   Top-1 vs top-2 kNN distance gap. Large gap = confident retrieval = reliable.
   Requires kNN k>=2 during scoring (pipeline passes this via metadata).

2. Normal coverage (R_coverage):
   How well the memory bank covers this query's distance. If d1 is within the
   normal validation distance distribution, coverage is high. If d1 is far
   beyond normal tail, this scale may lack support OR it's a true anomaly —
   coverage alone is ambiguous, so it's combined with other cues.

3. Local consistency (R_local):
   Spatial variance of the anomaly map in a local neighborhood. Low variance
   = smooth, coherent response = reliable. High variance = noise/boundary.

4. Illumination shift (R_light):
   Feature-space mean shift between test tile and training mean. Large shift
   = lighting change = appearance-sensitive scales are less reliable.

Fusion:
    A_fused(p) = sum_{s,l} softmax(R_{s,l}(p)) * norm(A_{s,l}(p))

The reliability cues are combined multiplicatively (each in [0,1]) then
softmax-normalized across (scale, layer) for each pixel.
"""

import torch
import torch.nn.functional as F


def fuse(
    raw_maps: dict,
    output_shape: tuple,
    config: dict,
    **kwargs,
) -> torch.Tensor:
    """Reliability-weighted multi-scale fusion.

    Args:
        raw_maps:  {(scale, layer): (H, W) torch.Tensor} upsampled distance maps
        output_shape: (H, W)
        config:   fusion config dict, may contain:
            tau_local (float): local consistency temperature, default 1.0
            tau_light (float): illumination temperature, default 1.0
            cue_weights (dict): weights for {margin, coverage, local, light}
        **kwargs:  metadata dict with optional keys:
            'margins':      {(scale, layer): (H, W) tensor} — retrieval margin maps
            'normal_dists': {(scale, layer): (P99, P50)} — normal validation stats
            'feat_means':   {(scale, layer): (D,) tensor} — train feature means
            'test_feat_means': {(scale, layer): (H, W, D) tensor} — test tile means

    Returns:
        (H, W) torch.Tensor fused anomaly map
    """
    metadata = kwargs.get('metadata', {})
    cue_cfg = config.get('cue_weights', {
        'margin': 1.0, 'coverage': 1.0, 'local': 1.0, 'light': 1.0
    })
    tau_local = config.get('tau_local', 1.0)
    tau_light = config.get('tau_light', 1.0)

    keys = list(raw_maps.keys())
    maps = torch.stack([raw_maps[k] for k in keys], dim=0)  # (S*L, H, W)

    # --- Normalize each map to [0, 1] for fair fusion ---
    maps_norm = _normalize_maps(maps)

    # --- Compute reliability per (scale, layer) ---
    reliability = torch.ones_like(maps)  # (S*L, H, W)

    # Cue 1: Retrieval margin
    if 'margins' in metadata and cue_cfg.get('margin', 1.0) > 0:
        w = cue_cfg['margin']
        margin_maps = torch.stack([
            metadata['margins'].get(k, torch.ones_like(raw_maps[k]))
            for k in keys
        ], dim=0)
        # Sigmoid to [0, 1]: high margin = high reliability
        r_margin = torch.sigmoid(margin_maps - margin_maps.mean())
        reliability = reliability * (r_margin ** w)

    # Cue 2: Normal coverage
    if 'normal_dists' in metadata and cue_cfg.get('coverage', 1.0) > 0:
        w = cue_cfg['coverage']
        for i, k in enumerate(keys):
            stats = metadata['normal_dists'].get(k)
            if stats is not None:
                p99, p50 = stats
                # If distance < p50, coverage = 1. If > p99, coverage drops.
                # Linear ramp between p50 and p99, then exponential decay beyond.
                d = raw_maps[k]
                r_cov = torch.where(
                    d <= p50,
                    torch.ones_like(d),
                    torch.exp(-(d - p50) / (p99 - p50 + 1e-8))
                )
                reliability[i] = reliability[i] * (r_cov ** w)

    # Cue 3: Local consistency
    if cue_cfg.get('local', 1.0) > 0:
        w = cue_cfg['local']
        r_local = _local_consistency(maps, tau_local)
        reliability = reliability * (r_local ** w)

    # Cue 4: Illumination shift
    if 'feat_means' in metadata and 'test_feat_means' in metadata and cue_cfg.get('light', 1.0) > 0:
        w = cue_cfg['light']
        r_light = _illumination_reliability(
            keys, metadata, tau_light, maps.device, maps.shape[1:]
        )
        if r_light is not None:
            reliability = reliability * (r_light ** w)

    # --- Softmax-weighted fusion ---
    # reliability: (S*L, H, W) -> softmax across S*L dimension
    weights = F.softmax(reliability, dim=0)  # (S*L, H, W)
    fused = (weights * maps_norm).sum(dim=0)  # (H, W)

    return fused


def _normalize_maps(maps: torch.Tensor) -> torch.Tensor:
    """Per-map min-max normalization to [0, 1]."""
    S, H, W = maps.shape
    flat = maps.reshape(S, -1)
    vmin = flat.min(dim=-1, keepdim=True).values
    vmax = flat.max(dim=-1, keepdim=True).values
    norm = (flat - vmin) / (vmax - vmin + 1e-8)
    return norm.reshape(S, H, W)


def _local_consistency(maps: torch.Tensor, tau: float) -> torch.Tensor:
    """Local consistency via spatial variance in a 3x3 neighborhood.

    Low local variance = high reliability.
    """
    S, H, W = maps.shape
    # Compute local variance using unfold
    pad = 1
    padded = F.pad(maps.unsqueeze(1), (pad, pad, pad, pad), mode='replicate')
    # (S, 1, H+2, W+2) -> (S, 9, H, W)
    unfolded = F.unfold(padded, kernel_size=3, stride=1)
    unfolded = unfolded.reshape(S, 9, H, W)
    local_var = unfolded.var(dim=1)  # (S, H, W)
    return torch.exp(-local_var / (tau + 1e-8))


def _illumination_reliability(
    keys: list,
    metadata: dict,
    tau: float,
    device: str,
    spatial_shape: tuple,
) -> torch.Tensor | None:
    """Illumination reliability from feature mean shift.

    For each (scale, layer), compare test tile feature mean to training mean.
    Large shift = low reliability.
    """
    train_means = metadata.get('feat_means', {})
    test_means = metadata.get('test_feat_means', {})

    S = len(keys)
    H, W = spatial_shape
    r_light = torch.ones(S, H, W, device=device)

    has_any = False
    for i, k in enumerate(keys):
        if k in train_means and k in test_means:
            has_any = True
            train_m = train_means[k]  # (D,)
            test_m = test_means[k]    # (H, W, D) or (D,)
            if test_m.ndim == 1:
                # Global mean shift
                shift = (test_m - train_m).norm().item()
                r = torch.tensor(1.0 / (1.0 + shift / tau), device=device)
                r_light[i] = r
            else:
                # Per-tile mean shift
                shift = (test_m - train_m).norm(dim=-1)  # (H, W)
                r_light[i] = torch.exp(-shift / tau)

    return r_light if has_any else None