import numpy as np
import torch
from torch_geometric.datasets import Twitch, Planetoid
from torch_geometric.utils import to_dense_adj


def load_dataset(_dataset):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.

    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "Twitch" , "Cora"

    :returns: Respective dataset
    """
    print(f"Loading {_dataset} dataset")
    if _dataset == "Twitch":
        return load_TwitchDataset()
    elif _dataset == "cora":
        return load_cora_dataset()
    else:
        raise NotImplementedError


def load_TwitchDataset():
    """
    Loads the Twitch Dataset. Checks if the dataset is downloaded or not.
    If the dataset is not downloaded then creates a foler with the name raw and downloads all files.

    By using these files loads the dataset and stores it in a folder called processed

    Returns:
        _type_: Edge Index tensor, Features, Labels, Training mask, Validation Mask, Test Mask
    """
    graph = Twitch(root="./ExplanationEvaluation/datasets/Twitch", name="EN")
    graph = graph.data
    # Add train and test masks
    num_nodes = graph.x.shape[0]
    master = np.arange(0, num_nodes)
    np.random.shuffle(master)
    train = torch.zeros(num_nodes)
    train[master[2000:]] = 1
    train_mask = train.bool()

    val = torch.zeros(num_nodes)
    val[master[2001:3000]] = 1
    val_mask = val.bool()

    test = torch.zeros(num_nodes)
    test[master[3001: num_nodes]] = 1
    test_mask = test.bool()

    return graph.edge_index, graph.x, graph.y.numpy(), train_mask, val_mask, test_mask


def load_cora_dataset():
    """Load a node dataset - cora

    :returns: np.arrays
    """

    dataset = Planetoid(
        root='./ExplanationEvaluation/datasets', name='Cora')
    data = dataset[0]

    adj = to_dense_adj(data.edge_index)[0].numpy()
    features = data.x.numpy()
    labels = np.eye(dataset.num_classes, dtype='uint8')[data.y.numpy()]
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    val_mask = data.val_mask.numpy()

    return data.edge_index, data.x, data.y.numpy(), train_mask, val_mask, test_mask
