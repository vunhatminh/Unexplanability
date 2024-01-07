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


prog_args = uex_configs.arg_parse().parse_args()

print("*** EXPERIMENT SETTINGS")

if prog_args.dataset == 'syn_prop_1':
    motif_size = 6
    offset = 0
    explain_range = range(300,780)
elif prog_args.dataset == 'syn_agg_1':
    motif_size = 6
    offset = 0
    explain_range = range(300,780)
elif prog_args.dataset == 'syn_prop_2':
    motif_size = 6
    offset = 0
    explain_range = range(300,780)
elif prog_args.dataset == 'syn_agg_2':
    motif_size = 6
    offset = 0
    explain_range = range(300,780)
elif prog_args.dataset == 'syn1':
    motif_size = 5
    offset = 0
    explain_range = range(300,700)
elif prog_args.dataset == 'syn3':
    motif_size = 9
    offset = 3
    explain_range = range(300,1020)
elif prog_args.dataset == 'syn4':
    motif_size = 6
    offset = 1
    explain_range = range(511,871)
elif prog_args.dataset == 'syn5':
    motif_size = 9
    offset = 7
    explain_range = range(511,1231)
else:
    print("Not valid dataset.")

print("Dataset: ", prog_args.dataset)
print("Input dim: ", prog_args.input_dim)
print("Perturb prob: ", prog_args.perturb_prob)
print("Explainer: ", prog_args.explainer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset, data, results_path = uex_utils.load_dataset(prog_args)

if prog_args.model == "GCN":
    model = GCN_4(dataset) #GCN with 4 layers
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

an_tp  = []
dan_tp = []
an_fp  = []

ae_tp  = []
dae_tp = []
ae_fp  = []

n_tp  = []
dn_tp = []
n_fp  = []

e_tp  = []
de_tp = []
e_fp  = []

node_gt_dict = get_node_gt_dict(explain_range, motif_size, offset)
edge_gt_dict = get_edge_gt_dict(explain_range, data.edge_index, motif_size, offset)

for node_idx in explain_range:
    if node_idx % 10 == 0:
        print("Explaining node: ", node_idx)
        
    if prog_args.explainer == 'GNNExplainer':
    #   All perturabtion method
        node_feat_mask, edge_mask = explainer_a.explain_node(node_idx, data.x, data.edge_index)
        node_feat_mask = node_feat_mask.cpu().detach().numpy()
        edge_score_mat = convert.to_scipy_sparse_matrix(data.edge_index, edge_mask)
        edge_score_mat = edge_score_mat + edge_score_mat.T
#         edge_mask = edge_mask.cpu().detach().numpy()

#         an_tp.append(TP_node_score(node_feat_mask, node_idx, motif_size, offset))
#         dan_tp.append(TP_node_discrete(node_feat_mask, node_idx, motif_size, offset))
#         an_fp.append(FP_node_score(node_feat_mask, node_idx, motif_size, offset))
#         ae_tp.append(TP_edge_score(edge_mask, data.edge_index, node_idx, motif_size, offset))
#         dae_tp.append(TP_edge_discrete(edge_mask, data.edge_index, node_idx, motif_size, offset))
#         ae_fp.append(FP_edge_score(edge_mask, data.edge_index, node_idx, motif_size, offset))
        dan_tp.append(TP_node_discrete_from_dict(node_feat_mask, node_idx, node_gt_dict))
        dae_tp.append(TP_edge_discrete_from_dict(edge_score_mat, node_idx, edge_gt_dict))

    #   Node only
        node_feat_mask, edge_mask = explainer_n.explain_node(node_idx, data.x, data.edge_index)
        node_feat_mask = node_feat_mask.cpu().detach().numpy()

#         n_tp.append(TP_node_score(node_feat_mask, node_idx, motif_size, offset))
        dn_tp.append(TP_node_discrete(node_feat_mask, node_idx, motif_size, offset))
#         n_fp.append(FP_node_score(node_feat_mask, node_idx, motif_size, offset))

    #   Edge only
        node_feat_mask, edge_mask = explainer_e.explain_node(node_idx, data.x, data.edge_index)
        edge_score_mat = convert.to_scipy_sparse_matrix(data.edge_index, edge_mask)
        edge_score_mat = edge_score_mat + edge_score_mat.T
#         edge_mask = edge_mask.cpu().detach().numpy()

#         e_tp.append(TP_edge_score(edge_mask, data.edge_index, node_idx, motif_size, offset))
#         de_tp.append(TP_edge_discrete(edge_mask, data.edge_index, node_idx, motif_size, offset))
#         e_fp.append(FP_edge_score(edge_mask, data.edge_index, node_idx, motif_size, offset))
        de_tp.append(TP_edge_discrete_from_dict(edge_score_mat, node_idx, edge_gt_dict))


#     column_names = ['an TP', 'd-an TP', 'an FP',
#                    'en TP', 'd-en TP', 'en FP',
#                    'n TP', 'd-n TP', 'n FP',
#                    'e TP', 'd-e TP', 'e FP']

#     result_dict = {column_names[0]: an_tp,
#                    column_names[1]: dan_tp,
#                    column_names[2]: an_fp,
#                    column_names[3]: ae_tp,
#                    column_names[4]: dae_tp,
#                    column_names[5]: ae_fp,
#                    column_names[6]: n_tp,
#                    column_names[7]: dn_tp,
#                    column_names[8]: n_fp,
#                    column_names[9]: e_tp,
#                    column_names[10]: de_tp,
#                    column_names[11]: e_fp}
        
    elif prog_args.explainer == 'Grad':
        
    #   All perturabtion method
        node_feat_mask, edge_mask = explainer_a.explain_node(node_idx, data.x, data.edge_index)
        node_feat_mask = torch.abs(node_feat_mask).cpu().detach().numpy()
        edge_score_mat = convert.to_scipy_sparse_matrix(data.edge_index, torch.abs(edge_mask))
        edge_score_mat = edge_score_mat + edge_score_mat.T
        
        dan_tp.append(TP_node_discrete_from_dict(node_feat_mask, node_idx, node_gt_dict))
        dae_tp.append(TP_edge_discrete_from_dict(edge_score_mat, node_idx, edge_gt_dict))
        
    else:
        print("Not valid explainer.")
        
if prog_args.explainer == 'GNNExplainer':
    column_names = ['an TP', 'en TP', 'n TP', 'e TP']

    result_dict = {column_names[0]: dan_tp,
                   column_names[1]: dae_tp,
                   column_names[2]: dn_tp,
                   column_names[3]: de_tp}
        
elif prog_args.explainer == 'Grad':
    column_names = ['gn TP', 'ge TP']

    result_dict = {column_names[0]: dan_tp,
                   column_names[1]: dae_tp}    
        

    
results = pd.DataFrame(result_dict)
print(results.describe())
result_file_name = uex_utils.gen_name(prog_args) + " " + prog_args.explainer
results.to_pickle(result_file_name)

