"""Generate ablation config files for RCMN.

Produces all 2^3 = 8 combinations of (memory, fusion, calibration) strategies:
  - 000: baseline (knn_dedup, single_scale, percentile) = SuperADD
  - 100: +CMC only
  - 010: +RSF only (mean_multi as stepping stone)
  - 001: +NCC only
  - 110: +CMC +RSF
  - 101: +CMC +NCC
  - 011: +RSF +NCC
  - 111: full RCMN

Usage:
    python -m industrial.rcmn.ablation_configs
"""

import json
from pathlib import Path

from .config import load_config, get_root


STRATEGY_COMBOS = [
    # (memory, fusion, calibration, scales, name)
    ("knn_dedup", "single_scale", "percentile", [1.0], "000_baseline"),
    ("coverage_aware", "single_scale", "percentile", [1.0], "100_cmc"),
    ("knn_dedup", "mean_multi", "percentile", [0.5, 1.0], "010_mean_multi"),
    ("knn_dedup", "single_scale", "component", [1.0], "001_ncc"),
    ("coverage_aware", "mean_multi", "percentile", [0.5, 1.0], "110_cmc_mean"),
    ("coverage_aware", "single_scale", "component", [1.0], "101_cmc_ncc"),
    ("knn_dedup", "mean_multi", "component", [0.5, 1.0], "011_mean_ncc"),
    ("coverage_aware", "reliability", "component", [0.5, 1.0], "111_full_rcmn"),
]


def generate_configs(output_dir: str | Path | None = None):
    """Generate all ablation config JSON files."""
    base_config = load_config()

    if output_dir is None:
        output_dir = get_root() / 'experiments' / 'ablation_configs'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for memory, fusion, calibration, scales, name in STRATEGY_COMBOS:
        config = json.loads(json.dumps(base_config))  # deep copy
        config['memory']['strategy'] = memory
        config['fusion']['strategy'] = fusion
        config['fusion']['scales'] = scales
        config['calibration']['strategy'] = calibration
        # Use separate model/result dirs per ablation
        config['data']['models_dir'] = f'./models_rcmn/{name}/'
        config['data']['results_dir'] = f'./outputs_rcmn/{name}/'

        config_path = output_dir / f'{name}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f'generated {config_path}')

    print(f'\n{len(STRATEGY_COMBOS)} configs in {output_dir}')


if __name__ == '__main__':
    generate_configs()