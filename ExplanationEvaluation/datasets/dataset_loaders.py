import numpy as np
import torch
from torch_geometric.datasets import Twitch, Planetoid
from torch_geometric.utils import to_dense_adj


# def load_FacebookPagePageDataset():
#     rawPath = os.path.dirname(
#         os.path.realpath(__file__))
#     dataset = FacebookPagePage(root=rawPath+"/facebook")
#     data = dataset.data
#     adj = to_dense_adj(data.edge_index)[0].numpy()
#     features = data.x.numpy()
#     n_graphs = adj.shape[0]
#     labels = np.eye(dataset.num_classes, dtype='uint8')[data.y.numpy()]
#     train_indices = np.arange(0, int(n_graphs*0.8))
#     val_indices = np.arange(int(n_graphs*0.8), int(n_graphs*0.9))
#     test_indices = np.arange(int(n_graphs*0.9), n_graphs)
#     train_mask = np.full((n_graphs), False, dtype=bool)
#     train_mask[train_indices] = True
#     val_mask = np.full((n_graphs), False, dtype=bool)
#     val_mask[val_indices] = True
#     test_mask = np.full((n_graphs), False, dtype=bool)
#     test_mask[test_indices] = True

#     return data.edge_index, features, labels, train_mask, val_mask, test_mask


# def _load_node_dataset(_dataset):
#     """Load a node dataset.

#     :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
#     :returns: np.array
#     """
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     path = dir_path + '/pkls/' + _dataset + '.pkl'
#     with open(path, 'rb') as fin:
#         adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(
#             fin)
#     labels = y_train
#     labels[val_mask] = y_val[val_mask]
#     labels[test_mask] = y_test[test_mask]

#     return adj, features, labels, train_mask, val_mask, test_mask


# def load_politifact_dataset():
#     rawPath = os.path.dirname(
#         os.path.realpath(__file__))
#     dataset = FNNDataset(root=rawPath, feature="content",
#                          empty=False, name="politifact", transform=ToUndirected())

#     graph = dataset.data
#     print(graph)
#     # Add train and test masks
#     num_nodes = graph.x.shape[0]
#     master = np.arange(0, num_nodes)
#     np.random.shuffle(master)
#     train = torch.zeros(num_nodes)
#     train[master[2000:]] = 1
#     train_mask = train.bool()

#     val = torch.zeros(num_nodes)
#     val[master[2001:3000]] = 1
#     val_mask = val.bool()

#     test = torch.zeros(num_nodes)
#     test[master[3001: num_nodes]] = 1
#     test_mask = test.bool()

#     return graph.edge_index, graph.x, graph.y.numpy(), train_mask, val_mask, test_mask
#     return adjs, features, labels, train_mask, val_mask, test_mask


def load_dataset(_dataset, skip_preproccessing=False, shuffle=True):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.

    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :returns: multiple np.arrays
    """
    print(f"Loading {_dataset} dataset")
    # if _dataset == "facebook":
    #     return load_FacebookPagePageDataset()
    # if _dataset == "webKB":
    #     return load_WebKBDataset()
    if _dataset == "Twitch":
        return load_TwitchDataset()
    elif _dataset == "cora":
        return load_cora_dataset()        
    # if _dataset == "politifact":
    #     return load_politifact_dataset()
    else:
        raise NotImplementedError


# def load_WebKBDataset():
#     dataset = WebKB(
#         "/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/datasets/WebKB", "cornell")
#     adjs = to_dense_adj(dataset.data.edge_index)[0]
#     labels = np.eye(dataset.num_classes, dtype='uint8')[
#         dataset.data.y.numpy()]

#     n_graphs = adjs.shape[0]
#     indices = np.arange(0, n_graphs)
#     prng = RandomState(183)
#     indices = prng.permutation(indices)

#     # Create masks
#     train_indices = np.arange(0, int(n_graphs*0.7))
#     val_indices = np.arange(int(n_graphs*0.7), int(n_graphs*0.8))
#     test_indices = np.arange(int(n_graphs*0.8), n_graphs)
#     train_mask = np.full((n_graphs), False, dtype=bool)
#     train_mask[train_indices] = True
#     val_mask = np.full((n_graphs), False, dtype=bool)
#     val_mask[val_indices] = True
#     test_mask = np.full((n_graphs), False, dtype=bool)
#     test_mask[test_indices] = True

#     return dataset.data.edge_index, dataset.data.x, dataset.data.y, train_mask, val_mask, test_mask


def load_TwitchDataset():
    graph = Twitch(root="/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/datasets/Twitch", name="EN")
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

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    adj = to_dense_adj(data.edge_index)[0].numpy()
    features = data.x.numpy()
    labels = np.eye(dataset.num_classes, dtype='uint8')[data.y.numpy()]
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    val_mask = data.val_mask.numpy()

    print(labels)
    print(data.y.numpy())

    #return adj, features, labels, train_mask, val_mask, test_mask    
    return data.edge_index, data.x, data.y.numpy(), train_mask, val_mask, test_mask
