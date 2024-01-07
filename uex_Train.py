import os
import networkx as nx
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as colors

# Set matplotlib backend to file writing
plt.switch_backend("agg")

from torch_geometric.utils import from_networkx

from tensorboardX import SummaryWriter

import uex_synthetic_structsim
import uex_featgen
import uex_gengraph
import uex_configs
import uex_utils

from uex_models import *

prog_args = uex_configs.arg_parse().parse_args()

print("*** EXPERIMENT SETTINGS")
print("Dataset: ", prog_args.dataset)
# print("Input dim: ", prog_args.input_dim)
# print("Perturb prob: ", prog_args.perturb_prob)
print("Model: ", prog_args.model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset, data, results_path = uex_utils.load_dataset(prog_args)

if prog_args.model == "GCN":
    if prog_args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        model = GCNNet(dataset) #GCN with 2 layers
    else:
        model = GCN_4(dataset) #GCN with 4 layers
elif prog_args.model == "GIN":
    if prog_args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        model = GINConvNet2(dataset)
    else:
        model = GINConvNet(dataset)
elif prog_args.model == "GAT":
    if prog_args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        model = GATNet(dataset) #GAT with 2 layers
    else:
        model = GATNet3(dataset)  #GAT with 3 layers
else:
    print("Not support this model")
    model = None
    
model.to(device)
data = data.to(device)

print("*** BEGIN TRAINING")
print("Number of epochs: ", prog_args.num_epochs)
print("Learning rate: ", prog_args.lr)
train_model(model, data, epochs=prog_args.num_epochs, lr=prog_args.lr)

model_name = uex_utils.gen_name(prog_args) # + ".pth"
save_model(model, model_name)

print("Model is saved at ", model_name)