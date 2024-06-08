import torch
from torch import nn
import pickle
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import norm

# from data import new_train_dataset, train_dataset
from HGATConv import SimpleHGATConv
from torch_geometric.nn import Linear

metadata = None
with open("data/train_dataset_metadata.pickle", "rb") as file:
    metadata = pickle.load(file)
    print("loaded metadata")

# new_train_dataset = None
# with open("data/new_train_dataset.pickle", "rb") as file:
#   new_train_dataset = pickle.load(file)
#  print("loaded new_train_dataset")
#    print(new_train_dataset)
torch.manual_seed(42)


class EdgeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.enc1 = SimpleHGATConv(
            hidden_channels,
            hidden_channels,
            4,
            metadata,
            22,
            concat=True,
            residual=True,
        )
        #    self.enc1_bn = norm.BatchNorm(hidden_channels * 4)
        self.enc2 = SimpleHGATConv(
            hidden_channels * 4,
            hidden_channels * 4,
            4,
            metadata,
            22,
            concat=False,
            residual=True,
        )

    def forward(self, x, edge_index, node_type, edge_attr, edge_type):
        x = self.enc1(x, edge_index, node_type, edge_attr, edge_type).relu()

        # x = self.enc1_bn(x)
        x = self.enc2(x, edge_index, node_type, edge_attr, edge_type)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z, edge_label_index):
        row, col = edge_label_index

        z = torch.cat([z[row], z[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, training=False):
        super().__init__()
        self.encoder = EdgeEncoder(hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels * 4)

    def forward(self, batch_data):
        x = batch_data.x
        edge_index = batch_data.edge_index
        node_type = batch_data.node_type
        edge_attr = batch_data.edge_attr
        edge_type = batch_data.edge_type
        edge_label_index = batch_data.edge_label_index
        z = self.encoder(x, edge_index, node_type, edge_attr, edge_type)

        return self.decoder(z, edge_label_index), z


