from typing import Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import to_undirected

# ----------------------------------
# PyG Data Preparation
# ----------------------------------


def build_pyg_data(x: torch.FloatTensor, edge_index: torch.LongTensor) -> Data:
    """Create a PyG Data object from features and edge index."""
    return Data(x=x, edge_index=edge_index)


def split_data(
    data: Data,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    undirected: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[Data, Data, Data]:
    """Split edges into train/val/test and move to device."""
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=undirected,
        split_labels=True,
    )
    train, val, test = transform(data)

    if device is not None:
        train = train.to(device)
        val = val.to(device)
        test = test.to(device)

    print(
        f"Data split: train edges={train.pos_edge_label_index.size(1)}(+ neg), "
        f"val={val.pos_edge_label_index.size(1)}, test={test.pos_edge_label_index.size(1)}"
    )
    return train, val, test
