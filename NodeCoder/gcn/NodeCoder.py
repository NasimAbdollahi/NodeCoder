import torch
from torch_geometric.nn import GCNConv
import pandas as pd


class NodeCoder_Model(torch.nn.Module):
    """
    NodeCoder model.
    """
    def __init__(self, args):
        super(NodeCoder_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        if self.args.centrality_feature:
            self.args.input_channels = max(pd.read_csv(self.args.train_features_path[0])['feature_id'])+2
        else:
            self.args.input_channels = max(pd.read_csv(self.args.train_features_path[0])['feature_id'])+1
        self.args.output_channels = max(pd.read_csv(self.args.train_target_path[0])[self.args.target_name[0]])+1
        if self.args.multi_task_learning:
            self.args.output_layers = [[self.args.output_layer_size] for iter in range(0, len(self.args.target_name))]
        else:
            self.args.output_layers = [[self.args.output_layer_size] for iter in range(0, 1)]
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        self.model = StackedGCN(self.args)
        """ to avoid possible weight leakage: """
        self.model = self.model.to(self.device)


class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        :input_channels: Number of features.
        :output_channels: Number of target features. 
        """
        super(StackedGCN, self).__init__()
        self.args = args
        self.setup_layers()
        torch.manual_seed(self.args.seed)

    def setup_layers(self):
        """
        Creating the layes based on the args.
        """
        self.layers = []
        self.args.input_layers = [self.args.input_channels] + self.args.input_layers
        self.args.output_layer = [self.args.output_layer_size, self.args.output_channels]
        for i, _ in enumerate(self.args.input_layers[:-1]):
            self.layers.append(GCNConv(self.args.input_layers[i], self.args.input_layers[i+1]))
        for out_iter in range(0, len(self.args.output_layers)):
            self.layers.append(GCNConv(self.args.output_layer[0], self.args.output_layer[1]))
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features, edge_features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        torch.manual_seed(self.args.seed)
        for i in range(0, len(self.args.input_layers)-1):
            features = torch.nn.functional.elu(self.layers[i](features, edges, edge_features))
        predictions = []
        for io in range(0, len(self.args.output_layers)):
            x_out = self.layers[io+i+1](features, edges, edge_features)
            x_out = torch.nn.functional.dropout(x_out, p=self.args.dropout, training=self.training)
            predictions.append(torch.nn.functional.log_softmax(x_out, dim=1))
        return predictions


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)