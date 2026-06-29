"""RCMN anomaly detection pipeline.

Orchestrates three pluggable strategy stages (memory, fusion, calibration)
on top of a feature-extraction backend (DINOv3 + patching, reused from SuperADD).

Architecture:
    - Backend (method-specific): feature extraction + kNN retrieval
    - Strategies (method-agnostic): memory subsampling, map fusion, mask calibration

When all three strategies are set to baseline ("knn_dedup", "single_scale",
"percentile"), the pipeline reproduces SuperADD's behavior exactly.

To port to another AD method:
    - Replace the backend (backbone, preprocessing, patching, retrieval)
    - Reuse the three strategy stages as-is (they are pure functions)
"""

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from industrial.model import PreProcessing, PatchedExecution, DinoV3Backbone
from industrial.nearest_neighbor import nearest_neighbors

from .memory import MEMORY_STRATEGIES
from .fusion import FUSION_STRATEGIES
from .calibration import CALIBRATION_STRATEGIES
from .debug import save_debug_images


class RCMNPipeline:
    """Config-driven anomaly detection pipeline with pluggable strategies.

    Config structure (see config_rcmn.json):
        backend:      feature extraction params (backbone, layers, patching)
        memory:       {strategy, max_database_size, subsampling_iterations}
        fusion:       {strategy, scales}
        calibration:  {strategy, threshold params, morphology params}
        data:         paths and dataset config
        debug:        bool, save intermediate results
    """

    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device

        # --- Backend (method-specific) ---
        backend_cfg = config['backend']
        self.layers = backend_cfg['layers']
        self.backbone = DinoV3Backbone(
            backend_cfg['backbone'], self.layers, device
        )
        self.patch_exec = PatchedExecution(
            backend_cfg['patch_size'],
            backend_cfg['patch_overlap'],
            self.backbone.model_patch_size,
        )
        self.base_resize_factor = backend_cfg['patch_size'] / 1024
        self.evaluation_downscale = backend_cfg['evaluation_downscale']
        brightness_aug = tuple(backend_cfg.get('brightness_augmentation', (1.0, 1.0)))

        # --- Scales ---
        self.scales = config['fusion']['scales']

        # Per-scale preprocessing (train = augmented, test = clean)
        self.train_preprocessing: dict[float, PreProcessing] = {}
        self.test_preprocessing: dict[float, PreProcessing] = {}
        for s in self.scales:
            rf = self.base_resize_factor * s
            self.train_preprocessing[s] = PreProcessing(device, rf, brightness_aug)
            self.test_preprocessing[s] = PreProcessing(device, rf, (1.0, 1.0))

        # --- Strategy selection (method-agnostic) ---
        self.memory_fn = MEMORY_STRATEGIES[config['memory']['strategy']]
        self.fusion_fn = FUSION_STRATEGIES[config['fusion']['strategy']]
        self.calibration_fn = CALIBRATION_STRATEGIES[config['calibration']['strategy']]

        # --- State (populated by train()) ---
        self.memory_banks: dict[tuple[float, int], torch.Tensor] = {}
        self.normal_stats: dict = {}
        self.debug = config.get('debug', False)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _clear_cache(self):
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        gc.collect()

    def _extract_features(self, image: torch.Tensor, scale: float, train: bool) -> list:
        """Run preprocessing + patching + backbone at a given scale.

        Returns:
            list of np.ndarray, one per layer, each (h, w, c)
        """
        x = image.to(self.device)[None]
        if train:
            x = self.train_preprocessing[scale](x)
        else:
            x = self.test_preprocessing[scale](x)
        return self.patch_exec(x, self.backbone)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(self, train_images: list[torch.Tensor]):
        """Build memory banks and calibration stats from normal images.

        1. Split images into prototype set (builds memory) and threshold set (calibrates).
        2. For each scale: extract features, subsample into memory bank.
        3. Run predict on threshold set to compute normal anomaly score distribution.
        4. Store threshold and other normal stats for calibration.
        """
        calib_cfg = self.config['calibration']
        threshold_fraction = calib_cfg['threshold_fraction']

        prototype_images = [
            t for i, t in enumerate(train_images) if i % threshold_fraction != 0
        ]
        threshold_images = [
            t for i, t in enumerate(train_images) if i % threshold_fraction == 0
        ]

        self._clear_cache()

        # --- Build memory banks per scale, per layer ---
        for scale in self.scales:
            prototype_embeddings: dict[int, list] = defaultdict(list)

            for x in tqdm(
                prototype_images,
                desc=f'processing prototype (scale={scale})',
                file=sys.stdout,
            ):
                prediction = self._extract_features(x, scale, train=True)
                for layer, embedding in zip(self.layers, prediction):
                    prototype_embeddings[layer].append(
                        np.asarray(
                            embedding.reshape(-1, embedding.shape[-1]),
                            dtype=np.float16,
                        )
                    )

            for layer in self.layers:
                embeddings = np.concatenate(
                    prototype_embeddings[layer], axis=0
                ).astype(np.float32)
                del prototype_embeddings[layer]

                memory_cfg = self.config['memory']
                subsampled = self.memory_fn(
                    embeddings,
                    memory_cfg['max_database_size'],
                    self.device,
                    iterations=memory_cfg.get('subsampling_iterations', 100),
                )
                self.memory_banks[(scale, layer)] = torch.as_tensor(subsampled).to(
                    self.device
                )
                self._clear_cache()

        # --- Calibrate threshold on normal validation images ---
        anomaly_maps = []
        for x in tqdm(
            threshold_images,
            desc='processing threshold train data',
            file=sys.stdout,
        ):
            anomaly_map, _ = self.predict(x)
            anomaly_maps.append(anomaly_map)

        threshold = (
            np.percentile(anomaly_maps, calib_cfg['threshold_percentile'])
            * calib_cfg['threshold_factor']
        )
        self.normal_stats = {'threshold': float(threshold)}
        print(f'auto-detected threshold {threshold:.3f}')

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def predict(self, image: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run full pipeline on a single test image.

        Returns:
            anomaly_map:   (H, W) float32 continuous anomaly scores
            binary_result: (H, W) uint8 binary mask {0, 255}
        """
        input_shape = image.shape
        assert len(input_shape) == 3 and input_shape[0] == 3

        output_shape = (
            input_shape[-2] // self.evaluation_downscale,
            input_shape[-1] // self.evaluation_downscale,
        )

        # --- Stage 1: per-scale, per-layer kNN distance maps ---
        raw_maps: dict[tuple[float, int], torch.Tensor] = {}

        for scale in self.scales:
            prediction = self._extract_features(image, scale, train=False)

            for layer, predicted_embedding in zip(self.layers, prediction):
                _, h, w, c = predicted_embedding.shape
                query = torch.as_tensor(predicted_embedding).reshape(h * w, c).to(
                    self.device
                )
                keys = self.memory_banks[(scale, layer)]
                dists, _ = nearest_neighbors(
                    query, keys, knn_neighbors=1, normalize=False
                )
                dists = dists.mean(dim=-1).reshape(h, w) / c
                raw_maps[(scale, layer)] = dists

        self._clear_cache()

        # --- Upsample all raw maps to output_shape ---
        upsampled_maps: dict[tuple[float, int], torch.Tensor] = {}
        for key, dists in raw_maps.items():
            up = torch.nn.functional.interpolate(
                dists[None, None].float(),
                size=output_shape,
                mode='bilinear',
                align_corners=False,
            )
            upsampled_maps[key] = up[0, 0]

        # --- Stage 2: fusion ---
        fused = self.fusion_fn(
            upsampled_maps, output_shape, self.config['fusion']
        )
        anomaly_map = fused.cpu().numpy().astype(np.float32)

        # --- Stage 3: calibration ---
        binary_result = self.calibration_fn(
            anomaly_map, self.normal_stats, self.config['calibration']
        )

        self._clear_cache()

        return anomaly_map, binary_result

    # ------------------------------------------------------------------ #
    #  Debug
    # ------------------------------------------------------------------ #

    def predict_with_debug(
        self,
        image: torch.Tensor,
        debug_dir: Path,
        basename: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like predict() but saves intermediate results to debug_dir."""
        input_shape = image.shape
        output_shape = (
            input_shape[-2] // self.evaluation_downscale,
            input_shape[-1] // self.evaluation_downscale,
        )

        raw_maps: dict[tuple[float, int], torch.Tensor] = {}
        for scale in self.scales:
            prediction = self._extract_features(image, scale, train=False)
            for layer, predicted_embedding in zip(self.layers, prediction):
                _, h, w, c = predicted_embedding.shape
                query = torch.as_tensor(predicted_embedding).reshape(h * w, c).to(
                    self.device
                )
                keys = self.memory_banks[(scale, layer)]
                dists, _ = nearest_neighbors(
                    query, keys, knn_neighbors=1, normalize=False
                )
                dists = dists.mean(dim=-1).reshape(h, w) / c
                raw_maps[(scale, layer)] = dists

        self._clear_cache()

        upsampled_maps: dict[tuple[float, int], torch.Tensor] = {}
        for key, dists in raw_maps.items():
            up = torch.nn.functional.interpolate(
                dists[None, None].float(),
                size=output_shape,
                mode='bilinear',
                align_corners=False,
            )
            upsampled_maps[key] = up[0, 0]

        fused = self.fusion_fn(upsampled_maps, output_shape, self.config['fusion'])
        anomaly_map = fused.cpu().numpy().astype(np.float32)
        binary_result = self.calibration_fn(
            anomaly_map, self.normal_stats, self.config['calibration']
        )

        save_debug_images(debug_dir, basename, upsampled_maps, fused, anomaly_map, binary_result)

        self._clear_cache()
        return anomaly_map, binary_result

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Path):
        """Save config + memory banks + trained stats to disk.

        Writes:
            {path}.json  — config and trained normal_stats
            {path}.npz   — memory banks keyed by '{scale}_{layer}'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'config': self.config,
            'trained': self.normal_stats,
        }
        json.dump(save_data, open(f'{path}.json', 'w'), indent=4)

        memory = {
            f'{scale}_{layer}': v.cpu().numpy()
            for (scale, layer), v in self.memory_banks.items()
        }
        np.savez(f'{path}.npz', **memory)

    @staticmethod
    def load(path: Path, device: str = 'cuda') -> 'RCMNPipeline':
        """Load pipeline from disk."""
        path = Path(path)
        data = json.load(open(f'{path}.json', 'r'))

        pipeline = RCMNPipeline(data['config'], device)
        pipeline.normal_stats = data['trained']

        memory = np.load(f'{path}.npz')
        pipeline.memory_banks = {}
        for k, v in memory.items():
            parts = k.rsplit('_', 1)
            scale = float(parts[0])
            layer = int(parts[1])
            pipeline.memory_banks[(scale, layer)] = torch.from_numpy(v).to(device)

        return pipeline