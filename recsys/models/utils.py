from typing import List, Tuple

import torch

from recsys.datasets.utils import feature
from recsys.layers import CategoricalLayer, NumericalLayer


def align_item_user_dimensions(
    user_features,
    item_features,
    feature_embedding_size,
    id_embedding_size,
):
    user_features_dimensions = (
        estimate_embedding_size(user_features, feature_embedding_size)
        - id_embedding_size
    )
    item_features_dimensions = (
        estimate_embedding_size(item_features, feature_embedding_size)
        - id_embedding_size
    )

    max_dimension = max(item_features_dimensions, user_features_dimensions)
    user_id_dimensions = id_embedding_size + (max_dimension - user_features_dimensions)
    item_id_dimensions = id_embedding_size + (max_dimension - item_features_dimensions)

    return user_id_dimensions, item_id_dimensions


def estimate_embedding_size(
    features: List[feature], feature_embedding_size: int
) -> int:
    """
    Estimate the embedding size for a list of features.

    Categorical features will be processed by embedding layers, while numerical
    will just return its same value.
    """
    feature_dimension = 0
    for feature in features.values():
        if feature.dtype == "category":
            feature_dimension += feature_embedding_size
        elif feature.dtype == "int64":
            feature_dimension += 1

    return feature_dimension


def schema_to_featureModuleList(
    features: List[feature], feature_embedding_size: int, sizes_dictionary
) -> Tuple[torch.nn.ModuleList, int]:
    """
    Based on a list of features, create a nn.ModuleList of needed layers to
    process each feature.

    Categorical features will be processed by embedding layers, while numerical
    will just return its same value.
    """
    features_module = torch.nn.ModuleList()
    feature_dimension = 0
    for feature_idx, feature in enumerate(features.values()):
        if feature.dtype == "category":
            layer_name = f"{feature.name}"
            dimension = sizes_dictionary.get(feature.name, feature_embedding_size)
            f_layer = CategoricalLayer(
                name=layer_name,
                n_unique_values=feature.unique_value_count,
                dimensions=dimension,
                idx=feature_idx,
            )
            features_module.append(f_layer)
            feature_dimension += dimension
        elif feature.dtype == "int64":
            layer_name = f"{feature.name}"
            f_layer = NumericalLayer(
                name=layer_name,
                idx=feature_idx,
            )
            features_module.append(f_layer)
            feature_dimension += 1

    return features_module, feature_dimension
