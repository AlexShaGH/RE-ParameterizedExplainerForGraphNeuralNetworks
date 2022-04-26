from torch_geometric.datasets import Twitch, Planetoid


def _load_Twitch_node_dataset_ground_truth():
    dataset = Twitch(
        "/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/datasets/Twitch", name="EN")
    data = dataset.data
    return data.edge_index.numpy(), data.y.numpy().astype("float64")

def _load_cora_node_dataset_ground_truth():

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    return data.edge_index.numpy(), data.y.numpy().astype("float64")    


def load_dataset_ground_truth(_dataset, test_indices=None):
    """Load a the ground truth from a dataset.
    Optionally we can only request the indices needed for testing.

    :param test_indices: Only return the indices used by the PGExplaier paper.
    :returns: (np.array, np.array), np.array
    """
    if _dataset == "Twitch":
        graph, labels = _load_Twitch_node_dataset_ground_truth()
        if test_indices is None:
            return (graph, labels), range(100, 128, 5)
        else:
            all = range(0, 128, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    elif _dataset == "cora":
        graph, labels = _load_cora_node_dataset_ground_truth()
        if test_indices is None:
            #TO DO
            return (graph, labels), range(100, 128, 5)
        else:
            #TO DO
            all = range(0, 128, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered            
    else:
        print("Dataset does not exist")
        raise ValueError
