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
import numpy as np

""" from https://github.com/funket/zorro/ """

# a slight adoption of the method of Planetoid
def create_train_val_test_mask(data, num_train_per_class=20, num_classes=None, num_val=500, num_test=1000, ):
    import numpy as np
    # fix seed for selecting train_mask
    rng = np.random.RandomState(seed=42 * 20200909)

    if num_classes is None:
        num_classes = torch.max(data.y)

    train_mask = torch.full_like(data.y, False, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero().view(-1)
        idx = idx[rng.permutation(idx.size(0))[:num_train_per_class]]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero().view(-1)
    remaining = remaining[rng.permutation(remaining.size(0))]

    val_mask = torch.full_like(data.y, False, dtype=torch.bool)
    val_mask[remaining[:num_val]] = True

    test_mask = torch.full_like(data.y, False, dtype=torch.bool)
    test_mask[remaining[num_val:num_val + num_test]] = True

    return train_mask, val_mask, test_mask


class GCNNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN_3(torch.nn.Module):
    # only for syn2
    def __init__(self, dataset):
        super(GCN_3, self).__init__()
        hidden_dim = 20
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.lin_pred = Linear(3 * hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_all = [x]
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = self.lin_pred(x)

        return F.log_softmax(x, dim=1)
    
class GCN_4(torch.nn.Module):
    # only for syn2
    def __init__(self, dataset):
        super(GCN_4, self).__init__()
        hidden_dim = 20
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.lin_pred = Linear(4 * hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_all = [x]
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = self.lin_pred(x)

        return F.log_softmax(x, dim=1)
    
class GCN_5(torch.nn.Module):
    # only for syn2
    def __init__(self, dataset):
        super(GCN_5, self).__init__()
        hidden_dim = 20
        self.conv1 = GCNConv(dataset.num_node_features, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.conv5 = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.lin_pred = Linear(5 * hidden_dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x_all = [x]
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x_all.append(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = self.lin_pred(x)

        return F.log_softmax(x, dim=1)




class GATNet(torch.nn.Module):
    # based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATNet3(torch.nn.Module):
    # based on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
    def __init__(self, dataset):
        super(GATNet3, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, 8 * 8, heads=8, concat=False,
                             dropout=0.6)
        self.conv3 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP2Net(torch.nn.Module):
    def __init__(self, dataset):
        super(APPNP2Net, self).__init__()
        # default values from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/citation/appnp.py
        self.dropout = 0.5
        self.hidden = 64
        self.K = 2  # adjusted to two layers
        self.alpha = 0.1
        self.lin1 = Linear(dataset.num_node_features, self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = APPNP(self.K, self.alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GINConvNet2(torch.nn.Module):
    def __init__(self, dataset):
        super(GINConvNet2, self).__init__()

        num_features = dataset.num_node_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class GINConvNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GINConvNet, self).__init__()

        num_features = dataset.num_node_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_model(path, model):
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(path))
    model.eval()


def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, clip=None, loss_function="nll_loss",
                epoch_save_path=None, no_output=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    accuracies = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if loss_function == "nll_loss":
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        elif loss_function == "cross_entropy":
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], size_average=True)
        else:
            raise Exception()
        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        loss.backward()
        optimizer.step()

        if epoch_save_path is not None:
            # circumvent .pt ending
            save_model(model, epoch_save_path[:-3] + "_epoch_" + str(epoch) + epoch_save_path[-3:])
            accuracies.append(retrieve_accuracy(model, data, value=True))
            print('Accuracy: {:.4f}'.format(accuracies[-1]), "Epoch", epoch)
        else:
            if epoch % 25 == 0 and not no_output:
                print(retrieve_accuracy(model, data))

    model.eval()

    return accuracies


def save_model(model, path):
    torch.save(model.state_dict(), path)


def retrieve_accuracy(model, data, test_mask=None, value=False):
    _, pred = model(data.x, data.edge_index).max(dim=1)
    if test_mask is None:
        test_mask = data.test_mask
    correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / test_mask.sum().item()
    if value:
        return acc
    else:
        return 'Accuracy: {:.4f}'.format(acc)