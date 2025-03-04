
from typing import Dict
import numpy as np

def aggregate_patch_features_all(
    features_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Aggregates patch features by computing mean, max, min, and std.

    Args:
        features_dict: Dictionary mapping sample IDs to patch feature arrays.

    Returns:
        A dictionary mapping sample IDs to aggregated feature vectors.
    """
    aggregated_features = {}

    for sample_id, patches in sorted(features_dict.items()):
        mean_features = np.mean(patches, axis=0)
        max_features = np.max(patches, axis=0)
        min_features = np.min(patches, axis=0)
        std_features = np.std(patches, axis=0)

        aggregated_vector = np.concatenate(
            [mean_features, max_features, min_features, std_features]
        )
        aggregated_features[sample_id] = aggregated_vector

    return aggregated_features

def aggregate_patch_features_mean(
    features_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Aggregates patch features by computing mean, max, min, and std.

    Args:
        features_dict: Dictionary mapping sample IDs to patch feature arrays.

    Returns:
        A dictionary mapping sample IDs to aggregated feature vectors.
    """
    aggregated_features = {}

    for sample_id, patches in sorted(features_dict.items()):
        mean_features = np.mean(patches, axis=0)
        aggregated_vector = np.concatenate(
            [mean_features]
        )
        aggregated_features[sample_id] = aggregated_vector

    return aggregated_features


def aggregate_patch_features_min_max(
    features_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Aggregates patch features by computing mean, max, min, and std.

    Args:
        features_dict: Dictionary mapping sample IDs to patch feature arrays.

    Returns:
        A dictionary mapping sample IDs to aggregated feature vectors.
    """
    aggregated_features = {}

    for sample_id, patches in sorted(features_dict.items()):
        max_features = np.max(patches, axis=0)
        min_features = np.min(patches, axis=0)

        aggregated_vector = np.concatenate(
            [ max_features, min_features]
        )
        aggregated_features[sample_id] = aggregated_vector

    return aggregated_features