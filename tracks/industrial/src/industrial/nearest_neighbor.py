import torch
import numpy as np
from tqdm import tqdm
import sys


def nearest_neighbors(features_query, features_key, knn_neighbors, normalize=False):

    if normalize:
        features_query = torch.nn.functional.normalize(features_query, dim=-1)
        features_key = torch.nn.functional.normalize(features_key, dim=-1)

    dists_full = torch.cdist(features_query, features_key, compute_mode='use_mm_for_euclid_dist')  # [Nq, Nk] Fast GEMM-based distance computation

    topk_vals, topk_idx = torch.topk(
        dists_full,
        k=knn_neighbors,
        dim=-1,
        largest=False,  # Set largest=False to get smallest distances
        sorted=False
    )
    return topk_vals, topk_idx


def subsampling_distance_based_fast(features, target_number_of_samples, device, iterations=100, normalize=False, knn_neighbors=100):
    def subsample(x, target_number_of_samples):
        x_torch = torch.from_numpy(x).to(device)
        dists, _ = nearest_neighbors(x_torch, x_torch, knn_neighbors=knn_neighbors, normalize=normalize)
        # Keep dists on GPU for fast threshold search (avoids CPU bottleneck on large matrices)
        target_distance_between_samples = dists.mean().item() / 10
        number_of_samples = target_number_of_samples + 1
        random_numbers = torch.rand(len(x), device=device)

        while number_of_samples > target_number_of_samples:
            subsampling_factor = (dists < target_distance_between_samples).sum(dim=-1).float() + 1
            keep_mask = random_numbers < (1 / subsampling_factor)
            number_of_samples = keep_mask.sum().item()
            target_distance_between_samples *= 1.1

        return keep_mask.cpu().numpy()

    # perform subsample iteratively on random subsets of the data to speed up the nearest neighbor search
    keep_mask_total = np.full(len(features), False)
    size_of_subsets = int(1 / iterations * len(features))  # Size of subsets
    target_to_keep_subset = target_number_of_samples // iterations  # Target number of samples to keep from each subset

    for i in range(iterations):

        candidate_indices = np.where(np.invert(keep_mask_total))[0]
        indices = np.random.choice(candidate_indices, size=min(size_of_subsets, len(candidate_indices)),
                                   replace=False)  # Randomly select a subset of candidates for this iteration
        keep_mask_subset = subsample(features[indices], target_to_keep_subset)

        keep_mask_total[indices] = keep_mask_subset  # Update the total keep mask with the results from this subset
        number_of_samples = np.sum(keep_mask_total)

    difference = target_number_of_samples - number_of_samples

    # Randomly select some of the kept samples to add back until we reach the target number of samples
    if difference > 0:
        indices_to_add_back = np.where(np.invert(keep_mask_total))[0]
        np.random.shuffle(indices_to_add_back)
        keep_mask_total[indices_to_add_back[:difference]] = True

    # do the subsampling
    features_subsampled = features[keep_mask_total]
    # print(f'finished subsampling from {len(features)} to {len(features_subsampled)} features')

    return features_subsampled
