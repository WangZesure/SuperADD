"""Single-scale fusion: mean across layers (SuperADD baseline).

Equivalent to SuperADD's approach of stacking layer distance maps and taking
the mean. When only one scale is used, this is identical to the baseline.
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
        config:   fusion config dict (unused for this strategy)
        **kwargs: optional metadata (unused)

    Returns:
        (H, W) torch.Tensor fused anomaly map
    """
    maps = list(raw_maps.values())
    stacked = torch.stack(maps, dim=0)
    return torch.mean(stacked, dim=0)