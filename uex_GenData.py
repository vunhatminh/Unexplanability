import os
import networkx as nx
import numpy as np
import math

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors
from torch_geometric.utils import from_networkx

import uex_gengraph
import uex_configs
import uex_featgen
import uex_utils

prog_args = uex_configs.arg_parse().parse_args()

print("*** EXPERIMENT SETTINGS ***")
print("Dataset: ", prog_args.dataset)
print("Input dim: ", prog_args.input_dim)
print("Perturb prob: ", prog_args.perturb_prob)

DATA_PATH = "data/" +  prog_args.dataset + "_" + str(prog_args.perturb_prob) + ".npz"

if prog_args.dataset is not None:
    if prog_args.dataset == "syn_prop_1":
        print("Generating syn_prop_1 dataset")

        G, labels, name = uex_gengraph.gen_propagate_mechanics_1(input_dim = prog_args.input_dim, 
                                                                 nb_shapes=80, width_basis=300, m=5, 
                                                                 perturb_prob = prog_args.perturb_prob)

        data = from_networkx(G)
        edge_index = data.edge_index.numpy()

        x = data.x.numpy().astype(np.float32)
        y = np.array(labels)

        train_ratio = prog_args.train_ratio

        num_nodes = x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        train_mask = np.full_like(y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        save_data = {"edge_index": edge_index,
                         "x": x,
                         "y": y,
                         "train_mask": train_mask,
                         "test_mask": test_mask,
                         "num_nodes": G.number_of_nodes()
                         }

        np.savez_compressed(DATA_PATH, **save_data)
        
    elif prog_args.dataset == "syn_prop_2":
        print("Generating syn_prop_2 dataset")

        G, labels, name = uex_gengraph.gen_propagate_mechanics_2(input_dim = prog_args.input_dim, 
                                                                 nb_shapes=80, width_basis=300, m=5, 
                                                                 perturb_prob = prog_args.perturb_prob)

        data = from_networkx(G)
        edge_index = data.edge_index.numpy()

        x = data.x.numpy().astype(np.float32)
        y = np.array(labels)

        train_ratio = prog_args.train_ratio

        num_nodes = x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        train_mask = np.full_like(y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        save_data = {"edge_index": edge_index,
                         "x": x,
                         "y": y,
                         "train_mask": train_mask,
                         "test_mask": test_mask,
                         "num_nodes": G.number_of_nodes()
                         }

        np.savez_compressed(DATA_PATH, **save_data)
    
    elif prog_args.dataset == "syn_agg_1":
        print("Generating syn_agg_1 dataset")

        G, labels, name = uex_gengraph.gen_aggregate_mechanics_1(input_dim = prog_args.input_dim, 
                                                                 nb_shapes=80, width_basis=300, m=5, 
                                                                 perturb_prob = prog_args.perturb_prob)

        data = from_networkx(G)
        edge_index = data.edge_index.numpy()

        x = data.x.numpy().astype(np.float32)
        y = np.array(labels)

        train_ratio = prog_args.train_ratio

        num_nodes = x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        train_mask = np.full_like(y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        save_data = {"edge_index": edge_index,
                         "x": x,
                         "y": y,
                         "train_mask": train_mask,
                         "test_mask": test_mask,
                         "num_nodes": G.number_of_nodes()
                         }

        np.savez_compressed(DATA_PATH, **save_data)
        
    elif prog_args.dataset == "syn_agg_2":
        print("Generating syn_agg_2 dataset")

        G, labels, name = uex_gengraph.gen_aggregate_mechanics_2(input_dim = prog_args.input_dim, 
                                                                 nb_shapes=80, width_basis=300, m=5, 
                                                                 perturb_prob = prog_args.perturb_prob)

        data = from_networkx(G)
        edge_index = data.edge_index.numpy()

        x = data.x.numpy().astype(np.float32)
        y = np.array(labels)

        train_ratio = prog_args.train_ratio

        num_nodes = x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        train_mask = np.full_like(y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        save_data = {"edge_index": edge_index,
                         "x": x,
                         "y": y,
                         "train_mask": train_mask,
                         "test_mask": test_mask,
                         "num_nodes": G.number_of_nodes()
                         }

        np.savez_compressed(DATA_PATH, **save_data)
        
    else:
        generate_function = "uex_gengraph.gen_" + prog_args.dataset

        G, labels, name =  eval(generate_function)(feature_generator=uex_featgen.ConstFeatureGen(np.ones(prog_args.input_dim, dtype=float)))

        data = from_networkx(G)
        edge_index = data.edge_index.numpy()

        x = data.x.numpy().astype(np.float32)
        y = np.array(labels)

        train_ratio = prog_args.train_ratio

        num_nodes = x.shape[0]
        num_train = int(num_nodes * train_ratio)
        idx = [i for i in range(num_nodes)]

        np.random.shuffle(idx)
        train_mask = np.full_like(y, False, dtype=bool)
        train_mask[idx[:num_train]] = True
        test_mask = np.full_like(y, False, dtype=bool)
        test_mask[idx[num_train:]] = True

        save_data = {"edge_index": edge_index,
                         "x": x,
                         "y": y,
                         "train_mask": train_mask,
                         "test_mask": test_mask,
                         "num_nodes": G.number_of_nodes()
                         }

        np.savez_compressed(DATA_PATH, **save_data) 
    
#     elif prog_args.dataset == "bitcoinalpha":
#         print("Loading bitcoinalpha dataset")
#         G, labels, name = utils.read_bitcoinalpha(feature_generator=None)
#         utils.save_XAL(G,labels,prog_args)
    
   
#     else:
#         print("Not support dataset.")

print("  ")
print("*** Test loading dataset ***")

dataset, data, results_path = uex_utils.load_dataset(prog_args)
print(dataset)
print(data)
