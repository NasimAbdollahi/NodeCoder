import torch
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from NodeCoder.utilities.utils import protein_clustering
from NodeCoder.utilities.config import logger

class Clustering(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, protein_filename, graph, features, edge_features, target, cluster_number: int = 1):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.cluster_number = cluster_number
        self.graph = graph
        self.features = features
        self.edge_features = edge_features
        self.target = target
        self.protein_filename = protein_filename
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1]
        self.class_count = np.max(self.target)+1
        # for muti-task learning, each label (target in each task) is 0 or 1

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "Physical":
            logger.info("Physical Protein Clustering: clustering is performed by grouping proteins.")
            self.physical_protein_clustering()
        else:
            logger.warning("Random Protein Clustering: clustering is performed by random graph clustering. Not recommended!")
            self.random_clustering()
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def physical_protein_clustering(self):
        """
        Clustering the graph by grouping proteins.
        """
        self.clusters = [cluster for cluster in range(self.cluster_number)]
        protein_filenames = np.array(pd.read_csv(self.protein_filename)["Protein File"])
        Proteins_Node = np.array(pd.read_csv(self.protein_filename)["Node Num"])
        self.cluster_membership = protein_clustering(protein_filenames, Proteins_Node, self.cluster_number)

    def general_data_partitioning(self):
        """
        Creating data partitions.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_edge_features = {}
        self.sg_targets = {}
        features = []
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] + [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            """ edge weight: """
            for j in range(0, len(self.edge_features)):
                if (self.edge_features[j, 0] == edge[0] and self.edge_features[j, 1] == edge[1] for edge in self.sg_edges[cluster]):
                    # features.append([self.edge_features[j, 2], self.edge_features[j, 3]])
                    # features.append(np.reciprocal(self.edge_features[j, 2])) #distance-Reciprocal
                    features.append(np.reciprocal(self.edge_features[j, 2])**2)  #distance-ReciprocalSquared
                    #features.append(np.reciprocal(self.edge_features[j, 3]))   #sequence distance
            self.sg_edge_features[cluster] = np.array(features)
            features = []

            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size=self.args.test_ratio)
            # self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            # Using all nodes for training (for now):
            self.sg_train_nodes[cluster].extend(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster], :]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster], :]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_edge_features[cluster] = torch.FloatTensor(self.sg_edge_features[cluster])
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])   
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])