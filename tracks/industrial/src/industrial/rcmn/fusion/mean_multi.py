"""Multi-scale mean fusion (ablation baseline).

Simple extension of single_scale to multiple scales: stack all (scale, layer)
maps and take the mean. No reliability weighting — serves as the naive
multi-scale baseline against which RSF's reliability fusion is compared.
"""

import torch


def fuse(
    raw_maps: dict,
    output_shape: tuple,
    config: dict,
    **kwargs,
) -> torch.Tensor:
    """Fuse raw maps by simple mean across all (scale, layer) entries.

    Args:
        raw_maps:  {(scale, layer): (H, W) torch.Tensor} all same size
        output_shape: (H, W) target spatial size (maps already upsampled)
        config:   fusion config dict (unused)
        **kwargs: optional metadata (unused)

    Returns:
        (H, W) torch.Tensor fused anomaly map
    """
    maps = list(raw_maps.values())
    stacked = torch.stack(maps, dim=0)
    return torch.mean(stacked, dim=0)