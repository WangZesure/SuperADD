"""Batch ablation runner: train + test all config combinations.

Usage:
    python -m industrial.rcmn.run_ablation
    python -m industrial.rcmn.run_ablation --configs experiments/ablation_configs/100_cmc.json
    python -m industrial.rcmn.run_ablation --skip_train  # only test existing models
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .config import get_root


def run_ablation(configs_dir: Path, skip_train: bool = False):
    """Run train + test for each config in configs_dir."""
    configs = sorted(configs_dir.glob('*.json'))
    if not configs:
        print(f'no configs found in {configs_dir}')
        print('generate them first: python -m industrial.rcmn.ablation_configs')
        return

    results = []

    for config_path in configs:
        name = config_path.stem
        print(f'\n{"="*60}')
        print(f'Running ablation: {name}')
        print(f'Config: {config_path}')
        print(f'{"="*60}')

        if not skip_train:
            print(f'\n--- Training {name} ---')
            ret = subprocess.run([
                sys.executable, '-m', 'industrial.rcmn.train',
                '--config', str(config_path),
            ])
            if ret.returncode != 0:
                print(f'TRAINING FAILED for {name}')
                results.append((name, 'TRAIN_FAILED', None))
                continue

        print(f'\n--- Testing {name} ---')
        ret = subprocess.run([
            sys.executable, '-m', 'industrial.rcmn.test',
            '--config', str(config_path),
        ])
        if ret.returncode != 0:
            print(f'TESTING FAILED for {name}')
            results.append((name, 'TEST_FAILED', None))
            continue

        # Read results
        config = __import__('json').load(open(config_path))
        results_dir = Path(config['data']['results_dir'])
        if not results_dir.is_absolute():
            results_dir = get_root() / results_dir
        scores_path = results_dir / 'scores_test_public.csv'
        if scores_path.exists():
            import pandas as pd
            df = pd.read_csv(scores_path)
            mean_row = df[df['Category'] == 'mean'].iloc[0]
            results.append((name, 'OK', {
                'AU': f'{mean_row["AU"]*100:.2f}',
                'F1': f'{mean_row["F1"]*100:.2f}',
            }))
        else:
            results.append((name, 'NO_SCORES', None))

    # Summary
    print(f'\n{"="*60}')
    print('Ablation Summary')
    print(f'{"="*60}')
    print(f'{"Config":<25} {"Status":<15} {"AU":>8} {"F1":>8}')
    print('-' * 60)
    for name, status, scores in results:
        if scores:
            print(f'{name:<25} {status:<15} {scores["AU"]:>8} {scores["F1"]:>8}')
        else:
            print(f'{name:<25} {status:<15} {"":>8} {"":>8}')

    # Save summary CSV
    summary_path = configs_dir.parent / 'results' / 'summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    rows = []
    for name, status, scores in results:
        row = {'Config': name, 'Status': status}
        if scores:
            row.update(scores)
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f'\nsummary saved to {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='Run RCMN ablation experiments')
    parser.add_argument('--configs', type=str, default=None,
                        help='Specific config file (single) or directory of configs')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, only test existing models')
    args = parser.parse_args()

    if args.configs:
        p = Path(args.configs)
        if p.is_file():
            # Single config: wrap in a temp dir
            import tempfile
            d = Path(tempfile.mkdtemp()) / p.parent
            d.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(p, d / p.name)
            run_ablation(d, skip_train=args.skip_train)
        else:
            run_ablation(p, skip_train=args.skip_train)
    else:
        configs_dir = get_root() / 'experiments' / 'ablation_configs'
        run_ablation(configs_dir, skip_train=args.skip_train)


if __name__ == '__main__':
    main()