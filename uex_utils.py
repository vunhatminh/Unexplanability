""" uex_utils.py
    Utilities for loading data.
"""

import os
import numpy as np
import pandas as pd
import scipy as sc
import pandas as pd
import networkx as nx
import pickle as pkl
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import GINConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import from_networkx
from pathlib import Path
from collections import namedtuple
import uex_featgen


from types import SimpleNamespace

def load_dataset(prog_args, working_directory = None):

    if prog_args.dataset == "Cora":
        dataset = Planetoid('data/Cora', name='Cora')
        data = dataset[0]
        results_path = "cora"
    elif prog_args.dataset == "CiteSeer":
        dataset = Planetoid('data/CiteSeer', name='CiteSeer')
        data = dataset[0]
        results_path = "citeseer"
    elif prog_args.dataset == "PubMed":
        dataset = Planetoid('data/PubMed', name='PubMed')
        data = dataset[0]
        results_path = "pubmed"
    elif prog_args.dataset == "Mutagenicity":
        data = gc_data(prog_args.dataset, 'data/Mutagenicity/Mutagenicity', 0.8)
        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)
        results_path = "mutagenicity"
    else:
        if working_directory is None:
            working_directory = Path(".").resolve()
        DATA_PATH = "data/" +  prog_args.dataset + "_" + str(prog_args.perturb_prob) + ".npz"
        try:
            save_data = np.load(working_directory.joinpath(DATA_PATH))
        except FileNotFoundError:
            print("File not found")

        transformed_data = {}
        for name in save_data:
            transformed_data[name] = torch.tensor(save_data[name])
        data = torch_geometric.data.Data.from_dict(transformed_data)

        results_path = prog_args.dataset

        Dataset = namedtuple("Dataset", "num_node_features num_classes")
        dataset = Dataset(data.x.shape[1], max(data.y.numpy()) + 1)

    return dataset, data, results_path

def gen_name(prog_args):
    return "results/" + str(prog_args.dataset) + "_" + str(prog_args.perturb_prob) + "_" + str(prog_args.model) 

def gc_data(dataset, dirname, train_ratio=0.8):
    """Process datasets made of multiple graphs 
    Args:
        dataset (str): name of the dataset considered 
        dirname (str): path to a folder 
        args_input_dim (int, optional): Number of features. Defaults to 10.
        args_train_ratio (float, optional): Train/val/test split. Defaults to 0.8.
    Returns:
        NameSpace: gathers all info about input dataset 
    """
    
    # Define path where dataset should be saved
    data_path = "data/{}.pth".format(dataset)

    # If already created, do not recreate
    if os.path.exists(data_path):
        data = torch.load(data_path)
    else:
        data = SimpleNamespace()
        with open('data/Mutagenicity/Mutagenicity.pkl', 'rb') as fin:
            data.edge_index, data.x, data.y = pkl.load(fin)

        # Define NumSpace dataset
        data.x = torch.FloatTensor(data.x)
        data.edge_index = torch.FloatTensor(data.edge_index)
        data.y = torch.LongTensor(data.y)
        _, data.y = data.y.max(dim=1)
        data.num_classes = 2
        data.num_features = data.x.shape[-1]
        data.num_nodes = data.edge_index.shape[1]
        data.num_graphs = data.x.shape[0]
        data.name = dataset

        # Shuffle graphs 
        p = torch.randperm(data.num_graphs)
        data.x = data.x[p]
        data.y = data.y[p]
        data.edge_index = data.edge_index[p]
        
        # Train / Val / Test split
        data.train_mask, data.val_mask, data.test_mask = split_function(
                        data.y, train_ratio)
        # Save data
        torch.save(data, data_path)
    return data

def selected_data(data, dataset):
    """ Select only mutagen graphs with NO2 and NH2
    Args:
        data (NameSpace): contains all dataset related info
        dataset (str): name of dataset
    Returns:
        NameSpace: subset of input data with only some selected graphs 
    """
    edge_lists, graph_labels, edge_label_lists, node_label_lists = \
            get_graph_data(dataset)
    # we only consider the mutagen graphs with NO2 and NH2.
    selected = []
    for gid in range(data.edge_index.shape[0]):
            if np.argmax(data.y[gid]) == 0 and np.sum(edge_label_lists[gid]) > 0:
                selected.append(gid)
    print('number of mutagen graphs with NO2 and NH2', len(selected))

    data.edge_index = data.edge_index[selected]
    data.x = data.x[selected]
    data.y = data.y[selected]
    data.edge_lists = [edge_lists[i] for i in selected]
    data.edge_label_lists = [edge_label_lists[i] for i in selected]
    data.selected = selected
    
    return data

def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
    """Performs stratified train/test/val split
    Args:
        total_size (int): dataset total number of instances
        y_true (numpy array): labels
        val_fraction (int): validation/test set proportion
        test_fraction (int): test and val sets proportion
        seed (int): seed value
    Returns:
        [torch.tensors]: train, validation and test masks - boolean values
    """
    # Split into a train, val and test set
    # Store indexes of the nodes belong to train, val and test set
    indexes = range(total_size)
    indexes_train, indexes_test = train_test_split(
        indexes, test_size=test_fraction, stratify=y_true, random_state=seed)
    indexes_train, indexes_val = train_test_split(indexes_train, test_size=val_fraction, stratify=y_true[indexes_train],
                                                  random_state=seed)
    # Init masks
    train_idxs = np.zeros(total_size, dtype=np.bool)
    val_idxs = np.zeros(total_size, dtype=bool)
    test_idxs = np.zeros(total_size, dtype=np.bool)

    # Update masks using corresponding indexes
    train_idxs[indexes_train] = True
    val_idxs[indexes_val] = True
    test_idxs[indexes_test] = True

    return torch.from_numpy(train_idxs), torch.from_numpy(val_idxs), torch.from_numpy(test_idxs)


def split_function(y, args_train_ratio=0.6, seed=10):
    return _get_train_val_test_masks(y.shape[0], y, (1-args_train_ratio)/2, (1-args_train_ratio), seed=seed)