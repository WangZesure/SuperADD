"""RCMN testing entry point.

Evaluates trained RCMN models on MVTec AD 2 test splits.
Computes pixel-level AU-ROC0.05 and SegF1 when ground truth is available
(test_public). For test_private / test_private_mixed, saves predictions only.

Usage:
    python -m industrial.rcmn.test
    python -m industrial.rcmn.test --config path/to/config_rcmn.json
"""

import argparse
import os.path
import sys

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torchvision import transforms
from tqdm import tqdm

from industrial.dataset import MVTecAD2Dataset

from .config import (
    load_config,
    get_dataset_path,
    get_model_path,
    get_result_csv_path,
    get_result_anomaly_images_path,
    get_result_anomaly_images_thresholded_path,
    get_debug_dir,
)
from .pipeline import RCMNPipeline


def main():
    parser = argparse.ArgumentParser(description='RCMN testing')
    parser.add_argument('--config', type=str, default=None, help='Path to config_rcmn.json')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = get_dataset_path(config)
    data_cfg = config['data']
    debug = config.get('debug', False)

    if debug:
        debug_dir = get_debug_dir(config)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for split in data_cfg['test_split']:
        print(f'\nprocessing split {split}')

        category_f1_scores = {}
        category_au_scores = {}

        for category in data_cfg['categories']:
            model_path = get_model_path(config, category)
            pipeline = RCMNPipeline.load(model_path)

            test_data = MVTecAD2Dataset(
                dataset_path, category, split, transform=transforms.ToTensor()
            )

            anomaly_maps = []
            binary_maps = []
            ground_truths = []

            for sample in tqdm(
                test_data,
                desc=f'processing {category} images',
                file=sys.stdout,
            ):
                basename = os.path.basename(sample.image_path).removesuffix('.png')

                if sample.label == 0 and not data_cfg['evaluate_good_images']:
                    continue

                if debug:
                    anomaly_map, binary_result = pipeline.predict_with_debug(
                        sample.image, debug_dir / category / split, basename
                    )
                else:
                    anomaly_map, binary_result = pipeline.predict(sample.image)

                anomaly_map = anomaly_map.astype(np.float16)

                if data_cfg['save_predictions']:
                    tiff.imwrite(
                        get_result_anomaly_images_path(config, category, split) / f'{basename}.tiff',
                        anomaly_map,
                    )
                    cv2.imwrite(
                        get_result_anomaly_images_thresholded_path(config, category, split) / f'{basename}.png',
                        binary_result,
                    )

                if sample.label == -1:
                    continue

                if sample.mask is None:
                    ground_truth = np.zeros_like(binary_result)
                else:
                    ground_truth = sample.mask.numpy().astype(np.uint8)[0]
                    ground_truth = cv2.resize(
                        ground_truth,
                        binary_result.shape[::-1],
                        interpolation=cv2.INTER_LINEAR,
                    )

                anomaly_maps.append(anomaly_map)
                binary_maps.append(binary_result)
                ground_truths.append(ground_truth)

            if len(ground_truths) > 0:
                anomaly_flat = np.array(anomaly_maps).ravel()
                binary_flat = np.array(binary_maps).ravel() > 0
                gt_flat = np.array(ground_truths).ravel() > 0

                roc_auc = roc_auc_score(gt_flat, anomaly_flat, max_fpr=0.05)
                f1 = f1_score(gt_flat, binary_flat)

                print(f'category {category}: {roc_auc*100:.2f}% (AU), {f1*100:.2f}% (F1)')
                category_au_scores[category] = roc_auc
                category_f1_scores[category] = f1

        if len(category_f1_scores) > 0:
            print(f'\n---------- Results -----------')
            print(f'{split}')
            results_df = pd.DataFrame(
                zip(
                    category_f1_scores.keys(),
                    category_au_scores.values(),
                    category_f1_scores.values(),
                ),
                columns=['Category', 'AU', 'F1'],
            )
            results_df.loc[len(results_df)] = [
                'mean',
                results_df['AU'].mean(),
                results_df['F1'].mean(),
            ]
            results_df.to_csv(get_result_csv_path(config, f'scores_{split}'))
            print(results_df)


if __name__ == '__main__':
    main()