from typing import List, Tuple

import torch

from torchrecsys.datasets.utils import feature
from torchrecsys.layers import CategoricalLayer, NumericalLayer


def schema_to_featureModuleList(
    features: List[feature], feature_embedding_size: int
) -> Tuple[torch.nn.ModuleList, int]:
    """
    Based on a list of features, create a nn.ModuleList of needed layers to
    process each feature.

    Categorical features will be processed by embedding layers, while numerical
    will just return its same value.
    """
    features_module = torch.nn.ModuleList()
    feature_dimension = 0
    for feature_idx, feature in enumerate(features):
        if feature.dtype == "category":
            layer_name = f"user_{feature.name}_embedding"
            f_layer = CategoricalLayer(
                name=layer_name,
                n_unique_values=feature.unique_value_count,
                dimensions=feature_embedding_size,
                idx=feature_idx,
            )
            features_module.append(f_layer)
            feature_dimension += feature_embedding_size
        elif feature.dtype == "int64":
            layer_name = f"user_{feature.name}_numerical"
            f_layer = NumericalLayer(
                name=layer_name,
                idx=feature_idx,
            )
            features_module.append(f_layer)
            feature_dimension += 1

    return features_module, feature_dimension
