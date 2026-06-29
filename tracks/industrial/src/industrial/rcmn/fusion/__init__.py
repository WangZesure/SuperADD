"""Multi-scale anomaly map fusion strategies.

A fusion strategy combines raw per-scale, per-layer anomaly maps into a single
continuous anomaly map. Interface (method-agnostic):

    fuse(raw_maps: dict, output_shape: tuple, config: dict, **kwargs) -> torch.Tensor

Args:
    raw_maps:      {(scale: float, layer: int): (H, W) torch.Tensor on GPU}
                   All maps are already upsampled to output_shape.
    output_shape:  (H, W) target spatial size
    config:        fusion config dict (contains 'scales', strategy-specific params)
    **kwargs:      optional metadata for reliability cues (added in M3)

Returns:
    fused anomaly map: (H, W) torch.Tensor on GPU

To add a new strategy:
    1. Create a new module (e.g. reliability.py)
    2. Implement the `fuse` function with the above signature
    3. Register it here in FUSION_STRATEGIES
"""

from . import single_scale, mean_multi, reliability

FUSION_STRATEGIES = {
    "single_scale": single_scale.fuse,
    "mean_multi": mean_multi.fuse,
    "reliability": reliability.fuse,
}