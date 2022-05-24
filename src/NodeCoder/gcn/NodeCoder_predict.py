import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from NodeCoder.utilities.config import logger


class NodeCoder_Predictor(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, NodeCoder, protein_graph_data, task_count, checkpoint_epoch):
        """
        :param args: Arguments object.
        :param train_clustered: clustered graph data of train set
        :param validation_clustered: clustered graph data of validation set
        """
        self.args = args
        self.task_count = task_count
        self.protein_graph_data = protein_graph_data
        self.prediction_checkpoint_epoch = checkpoint_epoch
        # self.TaskNum = np.shape(protein_graph_data.target)[1]
        self.TaskNum = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NodeCoder
        self.reset_weights()
        self.Optimizer()

        self.predictions = []
        self.predictions_prob = []
        self.targets = []

        self.Protein_BalancedAcc = []
        self.Protein_Precision = []
        self.Protein_Recall = []
        self.Protein_F1score = []
        self.Protein_ROCAUC = []
        self.Protein_PRAUC = []
        self.Protein_MCC = []

    def reset_weights(self):
        """
          Resetting model weights to avoid weight leakage in cross validation setting.
        """
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                logger.info(f"Reset trainable parameters of layer = {layer}")
                layer.reset_parameters()

    def Optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=self.args.betas)

    def do_prediction(self, graph_data, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        nodes = graph_data.sg_train_nodes[cluster].to(self.device)
        edges = graph_data.sg_edges[cluster].to(self.device)
        features = graph_data.sg_features[cluster].to(self.device)
        edge_features = graph_data.sg_edge_features[cluster].to(self.device)
        target = []
        nodes_0 = graph_data.sg_train_nodes[cluster]
        target_0 = np.array(graph_data.target[:, self.task_count])
        target.append(target_0[nodes_0])
        target = torch.LongTensor(target).to(self.device)
        predictions = self.model(edges, features, edge_features)
        node_count = nodes.shape[0]
        self.prepare_test_results(graph_data, target, predictions, cluster)
        return node_count

    def test(self):
        """
        Test is performed.
        """
        if self.args.multi_task_learning:
            CheckPoint_path = self.args.CheckPoint_path[0] + 'Model_CheckPoints_epoch' + str(self.prediction_checkpoint_epoch) + '.pt'
        else:
            CheckPoint_path = self.args.CheckPoint_path[self.task_count] + 'Model_CheckPoints_epoch' + str(self.prediction_checkpoint_epoch) + '.pt'
        checkpoint = torch.load(CheckPoint_path)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            logger.critical("saved trained model have a different architecture than what is set for inference.")
            logger.warning("if you trained NodeCoder with centrality_feature=True you will need to set centrality_feature=True"
                           " when running inefernce.")
            raise
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """ Print model's state_dict """
        # logger.info("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     logger.info(f"{param_tensor} \t {self.model.state_dict()[param_tensor].size()}")
        self.model.eval()
        self.test_evaluation()
        self.test_metrics_perprotein()

    def test_evaluation(self):
        """
        """
        self.protein_predictions = []
        self.protein_predictions_prob = []
        self.protein_targets = []
        self.protein_node_count_seen = 0
        self.protein_clusterCount = 0
        for cluster in self.protein_graph_data.clusters:
            self.protein_clusterCount += 1
            self.do_prediction(self.protein_graph_data, cluster)
        self.predictions.append(self.protein_predictions)
        self.predictions_prob.append(self.protein_predictions_prob)
        self.targets.append(self.protein_targets)

    def prepare_test_results(self, graph_data, target, predictions, cluster):
        """
        Preparing validation results for calculating metrics per task.
        """
        nodes = graph_data.sg_train_nodes[cluster].to(self.device)
        protein_predictions = []
        protein_predictions_prob = []
        for iter in range(0, self.TaskNum):
            protein_predictions.append(predictions[iter][nodes].argmax(1).detach().cpu().clone().numpy())
            predictions_prob_0 = torch.exp(predictions[iter][nodes]).detach().cpu().clone().numpy()
            protein_predictions_prob.append(predictions_prob_0[:, 1])
        if self.protein_clusterCount == 1:
            self.protein_predictions = torch.tensor(protein_predictions).to(self.device)
            self.protein_targets = target.to(self.device)
            self.protein_predictions_prob = torch.tensor(protein_predictions_prob).to(self.device)
        else:
            self.protein_predictions = torch.cat((self.protein_predictions, torch.tensor(protein_predictions).to(self.device)), 1).to(self.device)
            self.protein_targets = torch.cat((self.protein_targets, target.to(self.device)), 1).to(self.device)
            self.protein_predictions_prob = torch.cat((self.protein_predictions_prob, torch.tensor(protein_predictions_prob).to(self.device)), 1).to(self.device)

    def test_metrics_perprotein(self):
        """
        Scoring the test results per protein.
        """
        node_ID = np.array(pd.read_csv(self.args.protein_node_proteinID_path)["node_id"])
        protein_ID = np.array(pd.read_csv(self.args.protein_node_proteinID_path)["protein_id_flag"])
        Protein = []
        for i in range(0, max(protein_ID)+1):
            Protein.append(node_ID[np.where(protein_ID == i)])

            self.Protein_BalancedAcc.append(metrics.balanced_accuracy_score(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions[0, Protein[i]].cpu().reshape(-1, 1)))
            self.Protein_F1score.append(metrics.f1_score(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions[0, Protein[i]].cpu().reshape(-1, 1), pos_label=1,zero_division=1))
            self.Protein_MCC.append(metrics.matthews_corrcoef(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions[0, Protein[i]].cpu().reshape(-1, 1)))
            try:
                self.Protein_Precision.append(metrics.precision_score(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions[0, Protein[i]].cpu().reshape(-1, 1), zero_division=1)) #average="micro"
            except ValueError:
                self.Protein_Precision.append(0)
            try:
                self.Protein_Recall.append(metrics.recall_score(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions[0, Protein[i]].cpu().reshape(-1, 1), zero_division=1)) #average="micro", macro:This does not take label imbalance into account.
            except ValueError:
                self.Protein_Recall.append(0)
            try:
                self.Protein_ROCAUC.append(metrics.roc_auc_score(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions_prob[0, Protein[i]].cpu().reshape(-1, 1)))
            except ValueError:
                self.Protein_ROCAUC.append(0)
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(self.protein_targets[0, Protein[i]].cpu().reshape(-1, 1), self.protein_predictions_prob[0, Protein[i]].cpu().reshape(-1, 1))
                self.Protein_PRAUC.append(metrics.auc(recall, precision))
            except ValueError:
                self.Protein_PRAUC.append(0)