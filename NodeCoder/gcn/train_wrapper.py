from NodeCoder.graph_generator.clustering import Clustering
from NodeCoder.gcn.NodeCoder_train import NodeCoder_Trainer
from NodeCoder.utilities.utils import graph_reader, feature_reader, edge_feature_reader, target_reader, DownSampling, \
    optimum_epoch, csv_writter_performance_metrics, csv_writer_prediction

class Wrapper(object):
    """
    wrapper.
    """
    def __init__(self, args, NodeCoder_Network):
        self.args = args
        self.NodeCoder_Network = NodeCoder_Network

    def train_fold(self, i):
        train_graph = graph_reader(self.args.train_edge_path[i])
        train_features = feature_reader(self.args.train_features_path[i], self.args.train_edge_path[i], self.args.centrality_feature)
        train_edge_features = edge_feature_reader(self.args.train_edge_feature_path[i])
        train_target = target_reader(self.args.train_target_path[i], self.args.target_name)
        if self.args.downSampling_majority_class == 'Yes':
            train_graph, train_features, train_edge_features, train_target = DownSampling(self.args, train_graph, train_features, train_edge_features, train_target)
        train_clustered = Clustering(self.args, self.args.train_protein_filename_path[i], train_graph, train_features, train_edge_features, train_target, cluster_number=self.args.train_cluster_number)
        train_clustered.decompose()

        validation_graph = graph_reader(self.args.validation_edge_path[i])
        validation_edge_features = edge_feature_reader(self.args.validation_edge_feature_path[i])
        validation_features = feature_reader(self.args.validation_features_path[i], self.args.validation_edge_path[i], self.args.centrality_feature)
        validation_target = target_reader(self.args.validation_target_path[i], self.args.target_name)
        validation_clustered = Clustering(self.args, self.args.validation_protein_filename_path[i], validation_graph,
                                          validation_features, validation_edge_features, validation_target, cluster_number=self.args.validation_cluster_number)
        validation_clustered.decompose()

        trainer = NodeCoder_Trainer(self.args, self.NodeCoder_Network.model, train_clustered, validation_clustered, i)
        trainer.train()
        csv_writter_performance_metrics(trainer, i)
        """
        Now we find the optimum epoch and load the optimum trained model that is saved using checkpoints.
        Then run inference to regenerate the final predicted labels and calculate prediction scores per protein.
        """
        checkpoint_epoch = optimum_epoch(self.args.Metrics_path[i])
        inference = NodeCoder_Trainer(self.args, self.NodeCoder_Network.model, train_clustered, validation_clustered, i, checkpoint_epoch)
        inference.test()
        csv_writer_prediction(self.args.NodeCoder_usage, self.args.target_name, inference.validation_targets, inference.validation_predictions,
                              inference.validation_predictions_prob, self.args.validation_node_proteinID_path[i], self.args.Prediction_fileName[i])