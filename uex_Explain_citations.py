import os
import networkx as nx
import numpy as np
import math
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Set matplotlib backend to file writing
plt.switch_backend("agg")

from torch_geometric.utils import from_networkx
import torch_geometric.utils.convert as convert

from tensorboardX import SummaryWriter

import uex_synthetic_structsim
import uex_featgen
import uex_gengraph
import uex_configs
import uex_utils

from uex_models import *
from uex_evaluation_utils import *
from uex_gnn_explainer import GNNExplainer
from uex_grad_explainer import GradExplainer


def binarize_tensor(tensor, number_of_ones):
    binary_tensor = torch.zeros_like(tensor)
    _, top_indices = torch.topk(tensor, number_of_ones, sorted=False)
    binary_tensor[top_indices] = 1

    return binary_tensor

def distortion(model, node_index, feat_mat, edge_index, node_mask, samples = 20, random_seed = 12345, validity = False, device = 'cpu'):
    
    (num_nodes, num_features) = feat_mat.size()
   
    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)
    random_indices = torch.randint(num_nodes, 
                                   (samples, num_nodes, num_features),
                                   generator=rng,
                                   device=device,
                                   )
    random_indices = random_indices.type(torch.int64)
    
    mask = node_mask.repeat(num_features,1).T
    
    if validity:
        samples = 1
        feat_mat = torch.zeros_like(feat_mat)
    
    correct = 0.0
    
    predicted_label = model(x=feat_mat, edge_index=edge_index).argmax(dim=-1)[node_index]
    
    for i in range(samples):
        random_features = torch.gather(feat_mat,
                                        dim=0,
                                        index=random_indices[i, :, :])
        
        randomized_features =   mask * feat_mat + (1 - mask) * random_features
        
        log_logits = model(x=randomized_features, edge_index=edge_index)
        distorted_labels = log_logits.argmax(dim=-1)
        
        if distorted_labels[node_idx] == predicted_label:
            correct += 1
    
    return correct/samples


prog_args = uex_configs.arg_parse().parse_args()

print("*** EXPERIMENT SETTINGS")

print("Dataset: ", prog_args.dataset)
print("Explainer: ", prog_args.explainer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset, data, results_path = uex_utils.load_dataset(prog_args)

if prog_args.dataset == 'Cora':
    explain_range = range(data.x.shape[0])
    num_ones = 5
elif prog_args.dataset == 'CiteSeer':
    explain_range = range(300)
    num_ones = 5
elif prog_args.dataset == 'PubMed':
    explain_range = range(300)
    num_ones = 5
else:
    print("Not valid dataset.")

if prog_args.model == "GCN":
    model = GCNNet(dataset) #GCN with 2 layers
elif prog_args.model == "GIN":
    model = GINConvNet(dataset)
elif prog_args.model == "GAT":
    model = GATNet(dataset)
else:
    print("Not support this model")
    model = None
    
model.to(device)
data = data.to(device)

print("*** BEGIN LOADING")
model_name = uex_utils.gen_name(prog_args)
print("Loading model at ", model_name)
load_model(model_name, model)
print(retrieve_accuracy(model, data, test_mask=None, value=False))


if prog_args.explainer == 'GNNExplainer':
    explainer_a = GNNExplainer(model, epochs=400, return_type='log_prob', feat_mask_type = 'scalar', log = False, allow_edge_mask = True)
    explainer_n = GNNExplainer(model, epochs=400, return_type='log_prob', feat_mask_type = 'scalar', log = False, allow_edge_mask = False)
    explainer_e = GNNExplainer(model, epochs=400, return_type='log_prob', feat_mask_type = 'scalar', log = False, edge_only = True)
elif prog_args.explainer == 'Grad':
    explainer_a = GradExplainer(model, feat_mask_type = 'scalar', allow_edge_mask = True)
else:
    print("Not valid explainer.")

an_va = []
an_fi = []
ae_va = []
ae_fi = []

n_va = []
n_fi = []
e_va = []
e_fi = []

gn_va = []
gn_fi = []
ge_va = []
ge_fi = []

for node_idx in explain_range:
    if node_idx % 10 == 0:
        print("Explaining node: ", node_idx)
        
    if prog_args.explainer == 'GNNExplainer':
    #   All perturabtion method
        node_feat_mask, edge_mask = explainer_a.explain_node(node_idx, data.x, data.edge_index)
        
        node_binary_mask = binarize_tensor(node_feat_mask, num_ones)
        node_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        node_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)
        
        node_score_convert_dict = edge_to_node_score(data.edge_index, edge_mask, data.x.shape[0])
        node_score_convert = torch.tensor([node_score_convert_dict[i] for i in range(data.x.shape[0])])
        node_binary_mask = binarize_tensor(node_score_convert.to(device), num_ones)
        edge_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        edge_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)
        
        an_va.append(node_validity)
        an_fi.append(node_fidelity)
        ae_va.append(edge_validity)
        ae_fi.append(edge_fidelity)

    #   Node only
        node_feat_mask, edge_mask = explainer_n.explain_node(node_idx, data.x, data.edge_index)
        
        node_binary_mask = binarize_tensor(node_feat_mask, num_ones)
        node_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        node_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)

        n_va.append(node_validity)
        n_fi.append(node_fidelity)

    #   Edge only
        node_feat_mask, edge_mask = explainer_e.explain_node(node_idx, data.x, data.edge_index)
        
        node_score_convert_dict = edge_to_node_score(data.edge_index, edge_mask, data.x.shape[0])
        node_score_convert = torch.tensor([node_score_convert_dict[i] for i in range(data.x.shape[0])])
        node_binary_mask = binarize_tensor(node_score_convert.to(device), num_ones)
        edge_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        edge_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)
        
        e_va.append(edge_validity)
        e_fi.append(edge_fidelity)

    elif prog_args.explainer == 'Grad':
        
    #   All perturabtion method
        node_feat_mask, edge_mask = explainer_a.explain_node(node_idx, data.x, data.edge_index)

        node_binary_mask = binarize_tensor(node_feat_mask, num_ones)
        node_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        node_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)
        
        node_score_convert_dict = edge_to_node_score(data.edge_index, edge_mask, data.x.shape[0])
        node_score_convert = torch.tensor([node_score_convert_dict[i] for i in range(data.x.shape[0])])
        node_binary_mask = binarize_tensor(node_score_convert.to(device), num_ones)
        edge_validity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = True, device = device)
        edge_fidelity = distortion(model, node_idx, data.x, data.edge_index, node_binary_mask, validity = False, device = device, samples = 100)
        
        gn_va.append(node_validity)
        gn_fi.append(node_fidelity)
        ge_va.append(edge_validity)
        ge_fi.append(edge_fidelity)
        
    else:
        print("Not valid explainer.")
        
if prog_args.explainer == 'GNNExplainer':
    column_names = ['an va', 'ae va', 'n va', 'e va',
                   'an fi', 'ae fi', 'n fi', 'e fi']


    result_dict = {column_names[0]: an_va,
                   column_names[1]: ae_va,
                   column_names[2]: n_va,
                   column_names[3]: e_va,
                   column_names[4]: an_fi,
                   column_names[5]: ae_fi,
                   column_names[6]: n_fi,
                   column_names[7]: e_fi}
        
elif prog_args.explainer == 'Grad':
    column_names = ['gn va', 'ge va', 'gn fi', 'ge fi',]

    result_dict = {column_names[0]: gn_va,
                   column_names[1]: ge_va,
                   column_names[2]: gn_fi,
                   column_names[3]: ge_fi}  
        

    
results = pd.DataFrame(result_dict)
print(results.describe())
result_file_name = uex_utils.gen_name(prog_args) + " " + prog_args.explainer
results.to_pickle(result_file_name)

