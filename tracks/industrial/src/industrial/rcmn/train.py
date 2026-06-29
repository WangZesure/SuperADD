"""RCMN training entry point.

Usage:
    python -m industrial.rcmn.train
    python -m industrial.rcmn.train --config path/to/config_rcmn.json
"""

import argparse
import numpy as np
from torchvision import transforms

from industrial.dataset import MVTecAD2Dataset
from .config import load_config, get_dataset_path, get_model_path
from .pipeline import RCMNPipeline


def main():
    parser = argparse.ArgumentParser(description='RCMN training')
    parser.add_argument('--config', type=str, default=None, help='Path to config_rcmn.json')
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = get_dataset_path(config)

    for category in config['data']['categories']:
        print(f'processing category {category}')

        np.random.seed(42)

        train_data = MVTecAD2Dataset(
            dataset_path, category, 'train', transform=transforms.ToTensor()
        )

        train_images = [
            train_data[i].image
            for i in range(0, len(train_data), 1)
        ]

        pipeline = RCMNPipeline(config)
        pipeline.train(train_images)

        model_path = get_model_path(config, category)
        pipeline.save(model_path)
        print(f'saved model to {model_path}.npz')


if __name__ == '__main__':
    main()