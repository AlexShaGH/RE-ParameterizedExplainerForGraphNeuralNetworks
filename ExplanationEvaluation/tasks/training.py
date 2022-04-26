import os
import torch
import numpy as np
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.model_selector import model_selector
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from tqdm import tqdm


def evaluate(out, labels, _dataset):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    if _dataset == "facebook":
        mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4])
        mlb.fit(labels)
        labels = mlb.inverse_transform(labels)
        labels = torch.tensor(np.array(labels).squeeze())
    if _dataset == "Twitch":
        labels = labels.T[0]
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc


def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :retunrs: None
    """
    save_dir = f"/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/models/pretrained/{paper}/{dataset}"
    #save_dir = f"C:\\Users\\Alex\\Documents\\IIT\\CS579 - Online Social Network Analysis\\Project-2\\from shubhum\\RE-ParameterizedExplainerForGraphNeuralNetworks\\ExplanationEvaluation\\models\\pretrained\\{paper}\\{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # print("epoch : ", epoch, "  : saved at : ", save_dir)
    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :return: model with pramaters taken from the checkpoint
    """
    # print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(
            f"/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/models/pretrained/{paper}/{dataset}/best_model")
            #f"C:\\Users\\Alex\\Documents\\IIT\\CS579 - Online Social Network Analysis\\Project-2\\from shubhum\\RE-ParameterizedExplainerForGraphNeuralNetworks\\ExplanationEvaluation\\models\\pretrained\\{paper}\\{dataset}\\best_model")            
    else:
        checkpoint = torch.load(
            f"/Users/shubhammodi/Documents/CS 579/Project 2/RE-ParameterizedExplainerForGraphNeuralNetworks/ExplanationEvaluation/models/pretrained/{paper}/{dataset}/model_{best_epoch}")
            #f"C:\\Users\\Alex\\Documents\\IIT\\CS579 - Online Social Network Analysis\\Project-2\\from shubhum\\RE-ParameterizedExplainerForGraphNeuralNetworks\\ExplanationEvaluation\\models\\pretrained\\{paper}\\{dataset}\\model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled:
        model.eval()

    return model


def train_node(_dataset, _paper, args):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    edge_index, x, labels, train_mask, val_mask, test_mask = load_dataset(
        _dataset)
    model = model_selector(_paper, _dataset, False)

    # x = torch.tensor(features)
    labels = torch.tensor(labels, dtype=torch.long)
    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # if _dataset == "Twitch":
    #     criterion = torch.nn.BCEWithLogitsLoss()
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    train_results = []
    for epoch in tqdm(range(0, args.epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        # out = model(Data(x=x, edge_index=edge_index, y=torch.tensor(labels)))
        if _dataset == "Twitch":
            preds = out[train_mask]
            targets = torch.unsqueeze(labels[train_mask], 1)
            # print("preds : ", preds.shape)
            # print("targets : ", targets.shape)
            # loss = criterion(preds.float(), targets.float())
            loss = criterion(out[train_mask], labels[train_mask])
        else:
            # return out, labels
            loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled:
            model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            # out = model(Data(x=x, edge_index=edge_index,
            # y = torch.tensor(labels)))

        # Evaluate train
        if _dataset == "Twitch":
            train_acc = evaluate(preds, targets, _dataset)
            test_acc = evaluate(out[test_mask], torch.unsqueeze(
                labels[test_mask], 1), _dataset)
            val_acc = evaluate(out[val_mask], torch.unsqueeze(
                labels[val_mask], 1), _dataset)
        else:
            train_acc = evaluate(out[train_mask], labels[train_mask], _dataset)
            test_acc = evaluate(out[test_mask], labels[test_mask], _dataset)
            val_acc = evaluate(out[val_mask], labels[val_mask], _dataset)
        # print(
            # f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        train_results.append((train_acc, val_acc, loss))

        if val_acc > best_val_acc:  # New best results
            # print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc,
                             val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper, _dataset,
                            model, args.eval_enabled)
    out = model(x, edge_index)

    # Train eval
    if _dataset == "Twitch":
        train_acc = evaluate(preds, targets, _dataset)
        test_acc = evaluate(out[test_mask], torch.unsqueeze(
            labels[test_mask], 1), _dataset)
        val_acc = evaluate(out[val_mask], torch.unsqueeze(
            labels[val_mask], 1), _dataset)
    else:
        train_acc = evaluate(out[train_mask], labels[train_mask], _dataset)
        test_acc = evaluate(out[test_mask], labels[test_mask], _dataset)
        val_acc = evaluate(out[val_mask], labels[val_mask], _dataset)

    print(
        f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    df = pd.DataFrame(data=train_results,
                      columns=["Train Accuracy", "Validation Accuracy", "Training Loss"])

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)
    return df
