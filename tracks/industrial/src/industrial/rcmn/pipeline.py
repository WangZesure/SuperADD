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
        self.base_patch_size = backend_cfg['patch_size']
        self.base_patch_overlap = backend_cfg['patch_overlap']
        self.model_patch_size = self.backbone.model_patch_size
        self.base_resize_factor = backend_cfg['patch_size'] / 1024
        self.evaluation_downscale = backend_cfg['evaluation_downscale']
        brightness_aug = tuple(backend_cfg.get('brightness_augmentation', (1.0, 1.0)))

        # --- Scales ---
        # Each scale adjusts both resize_factor AND patch_size proportionally,
        # so the number of patches per image stays roughly constant.
        # scale=1.0 → original SuperADD behavior.
        # scale=0.5 → half resolution, half patch_size (global context).
        self.scales = config['fusion']['scales']

        self.patch_execs: dict[float, PatchedExecution] = {}
        self.train_preprocessing: dict[float, PreProcessing] = {}
        self.test_preprocessing: dict[float, PreProcessing] = {}
        for s in self.scales:
            rf = self.base_resize_factor * s
            ps = max(self.base_patch_size, int(self.base_patch_size * s))
            # Ensure divisibility by model_patch_size
            ps = (ps // self.model_patch_size) * self.model_patch_size
            po = max(self.model_patch_size, (self.base_patch_overlap * s) // self.model_patch_size * self.model_patch_size)
            self.patch_execs[s] = PatchedExecution(ps, po, self.model_patch_size)
            self.train_preprocessing[s] = PreProcessing(device, rf, brightness_aug)
            self.test_preprocessing[s] = PreProcessing(device, rf, (1.0, 1.0))

        # --- Strategy selection (method-agnostic) ---
        self.memory_fn = MEMORY_STRATEGIES[config['memory']['strategy']]
        self.fusion_fn = FUSION_STRATEGIES[config['fusion']['strategy']]
        self.calibration_fn = CALIBRATION_STRATEGIES[config['calibration']['strategy']]

        # --- State (populated by train()) ---
        self.memory_banks: dict[tuple[float, int], torch.Tensor] = {}
        # Initialize with placeholder threshold=0 so predict() works during
        # training's threshold calibration phase (before real threshold is set).
        # This mirrors SuperADD's self.threshold = 0 in __init__.
        self.normal_stats: dict = {'threshold': 0.0}
        # Reliability metadata (used by RSF fusion strategy)
        self.normal_dists: dict[tuple[float, int], tuple[float, float]] = {}
        self.feat_means: dict[tuple[float, int], torch.Tensor] = {}
        self.debug = config.get('debug', False)

    @property
    def _uses_reliability(self) -> bool:
        return self.config['fusion']['strategy'] == 'reliability'

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _clear_cache(self):
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        gc.collect()

    def _build_fusion_metadata(
        self, margin_maps: dict
    ) -> dict:
        """Build metadata dict for reliability fusion."""
        metadata = {}
        if self._uses_reliability and margin_maps:
            metadata['margins'] = margin_maps
        if self.normal_dists:
            metadata['normal_dists'] = self.normal_dists
        if self.feat_means:
            metadata['feat_means'] = self.feat_means
        return metadata

    def _compute_scale_support(
        self, upsampled_maps: dict
    ) -> np.ndarray | None:
        """Count how many (scale, layer) maps exceed threshold at each pixel.

        Used by NCC component calibration to filter single-scale false positives.
        Returns None when only one scale is used (no filtering benefit).
        """
        if len(self.scales) <= 1:
            return None
        threshold = self.normal_stats.get('threshold', 0)
        if threshold == 0:
            return None
        support = None
        for dists in upsampled_maps.values():
            binary = (dists > threshold).int()
            support = binary if support is None else support + binary
        return support.cpu().numpy().astype(np.int32) if support is not None else None

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
        return self.patch_execs[scale](x, self.backbone)

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
                # Store feature mean for illumination reliability cue
                self.feat_means[(scale, layer)] = (
                    self.memory_banks[(scale, layer)].mean(dim=0)
                )
                self._clear_cache()

        # --- Calibrate threshold on normal validation images ---
        # Also collect per-(scale, layer) distance stats for coverage reliability
        anomaly_maps = []
        normal_dist_values: dict[tuple[float, int], list[float]] = {
            k: [] for k in self.memory_banks
        }

        for x in tqdm(
            threshold_images,
            desc='processing threshold train data',
            file=sys.stdout,
        ):
            anomaly_map, _, raw_dists = self.predict(x, return_raw_dists=True)
            anomaly_maps.append(anomaly_map)
            if raw_dists is not None:
                for key, dists in raw_dists.items():
                    normal_dist_values[key].extend(dists)

        threshold = (
            np.percentile(anomaly_maps, calib_cfg['threshold_percentile'])
            * calib_cfg['threshold_factor']
        )
        self.normal_stats = {'threshold': float(threshold)}

        # Store normal distance distribution stats (P50, P99) for coverage cue
        for key, vals in normal_dist_values.items():
            if len(vals) > 0:
                self.normal_dists[key] = (
                    float(np.percentile(vals, 99)),
                    float(np.percentile(vals, 50)),
                )

        print(f'auto-detected threshold {threshold:.3f}')

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def predict(
        self, image: torch.Tensor, return_raw_dists: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict | None]:
        """Run full pipeline on a single test image.

        Returns:
            anomaly_map:   (H, W) float32 continuous anomaly scores
            binary_result: (H, W) uint8 binary mask {0, 255}
            raw_dists:     (optional) per-(scale,layer) flattened distance values,
                           used during training to build normal_dists stats
        """
        input_shape = image.shape
        assert len(input_shape) == 3 and input_shape[0] == 3

        output_shape = (
            input_shape[-2] // self.evaluation_downscale,
            input_shape[-1] // self.evaluation_downscale,
        )

        # --- Stage 1: per-scale, per-layer kNN distance maps ---
        # Use k=2 when reliability fusion is active (need margin cue)
        knn_k = 2 if self._uses_reliability else 1
        raw_maps: dict[tuple[float, int], torch.Tensor] = {}
        margin_maps: dict[tuple[float, int], torch.Tensor] = {}
        raw_dists_flat: dict[tuple[float, int], list[float]] | None = (
            {} if return_raw_dists else None
        )

        for scale in self.scales:
            prediction = self._extract_features(image, scale, train=False)

            for layer, predicted_embedding in zip(self.layers, prediction):
                _, h, w, c = predicted_embedding.shape
                query = torch.as_tensor(predicted_embedding).reshape(h * w, c).to(
                    self.device
                )
                keys = self.memory_banks[(scale, layer)]
                dists, _ = nearest_neighbors(
                    query, keys, knn_neighbors=knn_k, normalize=False
                )
                d1 = dists[:, 0].reshape(h, w) / c
                raw_maps[(scale, layer)] = d1

                if knn_k >= 2:
                    d2 = dists[:, 1].reshape(h, w) / c
                    margin_maps[(scale, layer)] = (d2 - d1) / (d1 + 1e-8)

                if raw_dists_flat is not None:
                    raw_dists_flat[(scale, layer)] = d1.flatten().cpu().tolist()

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
        metadata = self._build_fusion_metadata(margin_maps)
        fused = self.fusion_fn(
            upsampled_maps, output_shape, self.config['fusion'], metadata=metadata
        )
        anomaly_map = fused.cpu().numpy().astype(np.float32)

        # --- Stage 3: calibration ---
        # Compute scale_support for NCC if multiple scales
        scale_support = self._compute_scale_support(upsampled_maps)

        binary_result = self.calibration_fn(
            anomaly_map,
            self.normal_stats,
            self.config['calibration'],
            scale_support=scale_support,
        )

        self._clear_cache()

        return anomaly_map, binary_result, raw_dists_flat

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
        anomaly_map, binary_result, _ = self.predict(image)
        # Re-derive upsampled maps and fused for visualization
        # (predict already computed these but didn't return them)
        # For debug, we save what we can from the final outputs
        save_debug_images(
            debug_dir, basename, {}, None, anomaly_map, binary_result
        )
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
            'normal_dists': {
                f'{scale}_{layer}': list(stats)
                for (scale, layer), stats in self.normal_dists.items()
            },
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

        # Restore normal_dists stats
        for k, stats in data.get('normal_dists', {}).items():
            parts = k.rsplit('_', 1)
            scale = float(parts[0])
            layer = int(parts[1])
            pipeline.normal_dists[(scale, layer)] = tuple(stats)

        memory = np.load(f'{path}.npz')
        pipeline.memory_banks = {}
        for k, v in memory.items():
            parts = k.rsplit('_', 1)
            scale = float(parts[0])
            layer = int(parts[1])
            pipeline.memory_banks[(scale, layer)] = torch.from_numpy(v).to(device)
            # Restore feat_means for illumination cue
            pipeline.feat_means[(scale, layer)] = (
                pipeline.memory_banks[(scale, layer)].mean(dim=0)
            )

        return pipeline