"""Binary mask calibration strategies.

A calibration strategy converts a continuous anomaly map into a binary mask
suitable for SegF1 evaluation. Interface (method-agnostic):

    calibrate(anomaly_map: np.ndarray, normal_stats: dict, config: dict, **kwargs) -> np.ndarray

Args:
    anomaly_map:  (H, W) float32/float16 continuous anomaly scores
    normal_stats: dict with at least {'threshold': float}
                  (may contain more stats for advanced strategies)
    config:       calibration config dict (closing params, etc.)
    **kwargs:     optional metadata (component features, scale support, ...)

Returns:
    binary mask: (H, W) uint8 with values {0, 255}

To add a new strategy:
    1. Create a new module (e.g. component.py)
    2. Implement the `calibrate` function with the above signature
    3. Register it here in CALIBRATION_STRATEGIES
"""

from . import percentile

CALIBRATION_STRATEGIES = {
    "percentile": percentile.calibrate,
}