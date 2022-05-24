import torch
import random
import numpy as np
import pandas as pd
import math
import time
import os
from sklearn import metrics
from NodeCoder.utilities.utils import label_distribution, optimum_epoch
import warnings
from NodeCoder.utilities.config import logger


class NodeCoder_Trainer(object):
    """ training a StackedGCN on multiple clusters. """
    def __init__(self, args, NodeCoder, train_clustered, validation_clustered, fold, check_point: int = 0):
        """
        :param args: Arguments object.
        :param train_clustered: clustered graph data of train set
        :param validation_clustered: clustered graph data of validation set
        """
        self.args = args
        self.fold = fold
        self.TaskNum = np.shape(train_clustered.target)[1]
        self.train_clustered = train_clustered
        self.validation_clustered = validation_clustered
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = NodeCoder
        self.reset_weights()
        self.Optimizer()

        """ used for test only """
        self.predictions = []
        self.predictions_prob = []
        self.targets = []

        self.Train_Loss = []
        self.Train_Acc = []
        self.Train_BalancedAcc = []
        self.Train_Precision = []
        self.Train_Recall = []
        self.Train_F1score = []
        self.Train_ROCAUC = []
        self.Train_PRAUC = []
        self.Train_AvePrecision = []
        self.Train_Task_Loss = []
        self.Train_Task_Acc = []
        self.Train_Task_BalancedAcc = []
        self.Train_Task_Precision = []
        self.Train_Task_Recall = []
        self.Train_Task_F1score = []
        self.Train_Task_ROCAUC = []
        self.Train_Task_PRAUC = []

        self.Validation_Loss = []
        self.Validation_Acc = []
        self.Validation_BalancedAcc = []
        self.Validation_Precision = []
        self.Validation_Recall = []
        self.Validation_F1score = []
        self.Validation_ROCAUC = []
        self.Validation_PRAUC = []
        self.Validation_AvePrecision = []
        self.Validation_MCC = []
        self.Validation_Task_Loss = []
        self.Validation_Task_Acc = []
        self.Validation_Task_BalancedAcc = []
        self.Validation_Task_Precision = []
        self.Validation_Task_Recall = []
        self.Validation_Task_F1score = []
        self.Validation_Task_ROCAUC = []
        self.Validation_Task_PRAUC = []

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

    def do_forward_pass(self, cluster, epoch):
        """
        Making a forward pass with data from a given partition.
        Code can perform multi-task learning.
        :param cluster: Cluster index.
        :return average_loss: Average loss of all tasks on the cluster.
        :return node_count: Number of nodes.
        targets and predictions of clusters are concatenated for each task,
        They will be used later for calculating metrics.
        """
        #macro_nodes = self.train_clustered.sg_nodes[cluster].to(self.device)
        train_nodes = self.train_clustered.sg_train_nodes[cluster].to(self.device)
        features = self.train_clustered.sg_features[cluster].to(self.device)
        edges = self.train_clustered.sg_edges[cluster].to(self.device)
        edge_features = self.train_clustered.sg_edge_features[cluster].to(self.device)
        target = []
        train_nodes_0 = self.train_clustered.sg_train_nodes[cluster]
        for i in range(0, self.TaskNum):
            target_0 = np.array(self.train_clustered.target[:, i])
            target.append(target_0[train_nodes_0])
        target = torch.LongTensor(target).to(self.device)
        predictions = self.model(edges, features, edge_features)
        average_loss, task_loss = self.evaluate_loss(predictions, target, train_nodes)
        node_count = train_nodes.shape[0]
        if epoch in self.Performance_epochs:
            self.prepare_train_results(target, predictions, task_loss, cluster)
        return average_loss, task_loss, node_count

    def evaluate_loss(self, predictions, target, nodes):
        """
        Calculating loss for each task and sum for multi-task learning.
        :param nodes: Number of train/validation nodes in currently processed cluster.
        :return average_loss: accumulated loss over tasks in cluster (or batch).
        accumulated_loss_task will be used to evaluate the model performance for each task
        """
        average_loss = 0
        task_loss = np.zeros(self.TaskNum)
        for i in range(0, self.TaskNum):
            """ log weight ratio of majority class to minority class: """
            if self.MajorityClass[i] == 0:
                if self.args.weighted_loss == 'non':
                    weight = torch.tensor([1.0, 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Logarithmic':
                    weight = torch.tensor([1.0, math.log(self.train_ClassDistRatio[i])]).to(self.device)
                elif self.args.weighted_loss == 'Power_Logarithmic':
                    weight = torch.tensor([1.0, math.log(self.train_ClassDistRatio[i])**2]).to(self.device)
                elif self.args.weighted_loss == 'Sigmoid':
                    weight = torch.tensor([1/(1 + math.exp(-self.train_ClassDistRatio[i])), 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Linear':
                    weight = torch.tensor([1.0, self.train_ClassDistRatio[i]]).to(self.device)
                elif self.args.weighted_loss == 'Smoothed_Linear':
                    weight = torch.tensor([1.0, math.sqrt(self.train_ClassDistRatio[i])]).to(self.device)
            else:
                if self.args.weighted_loss == 'non':
                    weight = torch.tensor([1.0, 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Logarithmic':
                    weight = torch.tensor([math.log(self.train_ClassDistRatio[i]), 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Power_Logarithmic':
                    weight = torch.tensor([math.log(self.train_ClassDistRatio[i])**2, 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Sigmoid':
                    weight = torch.tensor([1.0, 1/(1 + math.exp(-self.train_ClassDistRatio[i]))]).to(self.device)
                elif self.args.weighted_loss == 'Linear':
                    weight = torch.tensor([self.train_ClassDistRatio[i], 1.0]).to(self.device)
                elif self.args.weighted_loss == 'Smoothed_Linear':
                    weight = torch.tensor([math.sqrt(self.train_ClassDistRatio[i]), 1.0]).to(self.device)

            loss_function = torch.nn.NLLLoss(weight=weight, reduction='mean')
            loss = loss_function(predictions[i][nodes], target[i][:])
            average_loss = average_loss + loss
            task_loss[i] = loss
        return average_loss, task_loss

    def update_average_train_loss(self, batch_average_loss, batch_task_loss, node_count):
        """
        Updating (accumulating) the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count: Number of nodes in currently processed cluster.
        """
        self.accumulated_train_loss = self.accumulated_train_loss + batch_average_loss.item()*node_count
        self.accumulated_train_task_loss = self.accumulated_train_task_loss + batch_task_loss*node_count
        self.train_node_count_seen = self.train_node_count_seen + node_count
        return

    def update_average_validation_loss(self, batch_average_loss, batch_task_loss, node_count):
        """
        Updating (accumulating) the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count: Number of nodes in currently processed cluster.
        """
        self.accumulated_validation_loss = self.accumulated_validation_loss + batch_average_loss.item()*node_count
        self.accumulated_validation_task_loss = self.accumulated_validation_task_loss + batch_task_loss*node_count
        self.validation_node_count_seen = self.validation_node_count_seen + node_count
        return

    def do_prediction(self, validation_clustered, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        #macro_nodes = validation_clustered.sg_nodes[cluster].to(self.device)
        validation_nodes = validation_clustered.sg_train_nodes[cluster].to(self.device)
        edges = validation_clustered.sg_edges[cluster].to(self.device)
        features = validation_clustered.sg_features[cluster].to(self.device)
        edge_features = validation_clustered.sg_edge_features[cluster].to(self.device)
        target = []
        validation_nodes_0 = validation_clustered.sg_train_nodes[cluster]
        for i in range(0, self.TaskNum):
            target_0 = np.array(validation_clustered.target[:, i])
            target.append(target_0[validation_nodes_0])
        target = torch.LongTensor(target).to(self.device)
        predictions = self.model(edges, features, edge_features)
        average_loss, task_loss = self.evaluate_loss(predictions, target, validation_nodes)
        node_count = validation_nodes.shape[0]
        self.prepare_validation_results(validation_clustered, target, predictions, cluster)
        return average_loss, task_loss, node_count

    def train(self):
        """
        Training the model.
        Train predictions and targets are prepared for performance evaluation: calculating metrics for all tasks.
        """
        self.train_ClassDistRatio, self.MajorityClass = label_distribution(self.train_clustered, self.args.target_name, "train")
        label_distribution(self.validation_clustered, self.args.target_name, "validation")

        if self.args.epochs % self.args.performance_step == 0:
           self.Performance_epochs = np.append(np.arange(0, self.args.epochs, self.args.performance_step), self.args.epochs-1)
        else:
            self.Performance_epochs = np.arange(0, self.args.epochs, self.args.performance_step)
        if self.args.epochs % self.args.checkpoint_step == 0:
            self.CheckPoint_epochs = np.append(np.arange(0+self.args.checkpoint_step, self.args.epochs, self.args.checkpoint_step), self.args.epochs-1)
        else:
            self.CheckPoint_epochs = np.arange(0+self.args.checkpoint_step, self.args.epochs, self.args.checkpoint_step)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=self.args.betas)
        self.model.train()
        """ Print train model's state_dict """
        # logger.info("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     logger.info(f"{param_tensor} \t {self.model.state_dict()[param_tensor].size()}")

        pe = 0
        epochs = np.arange(0, self.args.epochs)
        for epoch in epochs:
            start_time = time.time()
            random.shuffle(self.train_clustered.clusters)
            self.train_node_count_seen = 0
            self.accumulated_train_loss = 0
            self.accumulated_train_task_loss = np.zeros(self.TaskNum)
            self.clusterCount = 0
            self.train_predictions = []
            self.train_predictions_prob = []
            self.train_targets = []

            for cluster in self.train_clustered.clusters:
                self.clusterCount += 1
                self.optimizer.zero_grad()
                batch_average_loss, batch_task_loss, node_count = self.do_forward_pass(cluster, epoch)
                batch_average_loss.backward()
                self.optimizer.step()
                self.update_average_train_loss(batch_average_loss, batch_task_loss, node_count)
            if epoch in self.Performance_epochs:
                self.train_metrics()
                self.validation()
                logger.info(f"Epoch: {epoch+1:03d}, train loss: {self.Train_Loss[pe]:.6f}, validation loss:{self.Validation_Loss[pe]:.6f},"
                            f" train balanced acc: {self.Train_BalancedAcc[pe]:.6f}, validation balanced acc:{self.Validation_BalancedAcc[pe]:.6f},"
                            f" train roc-auc: {self.Train_ROCAUC[pe]:.6f}, validation roc-auc:{self.Validation_ROCAUC[pe]:.6f}")
                pe += 1
            if epoch in self.CheckPoint_epochs:
                if not os.path.exists(self.args.CheckPoint_path[self.fold]):
                    os.makedirs(self.args.CheckPoint_path[self.fold], exist_ok=True)
                CheckPoint_path = self.args.CheckPoint_path[self.fold] + 'Model_CheckPoints_epoch' + str(epoch) + '.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'Train_ROCAUC': self.Train_ROCAUC,
                    'Train_PRAUC': self.Train_PRAUC,
                    'Train_Loss': self.Train_Loss,
                    'Validation_ROCAUC': self.Validation_ROCAUC,
                    'Validation_PRAUC': self.Validation_ROCAUC,
                    'Validation_Loss': self.Validation_ROCAUC,
                }, CheckPoint_path)
            logger.info(f"time per epoch: {time.time() - start_time} seconds")

    def validation(self):
        """
        Validation is performed per epoch, which wil be performed for all clusters.
        Validation predictions and targets are prepared for evaluation: calculating metrics for all tasks
        """
        self.model.eval()
        self.validation_predictions = []
        self.validation_predictions_prob = []
        self.validation_predictions_prob_PR = []
        self.validation_targets = []
        self.validation_node_count_seen = 0
        self.accumulated_validation_loss = 0
        self.accumulated_validation_task_loss = np.zeros(self.TaskNum)
        self.validation_clusterCount = 0
        for cluster in self.validation_clustered.clusters:
            self.validation_clusterCount += 1
            average_loss, task_loss, test_node_count = self.do_prediction(self.validation_clustered, cluster)
            self.update_average_validation_loss(average_loss, task_loss, test_node_count)
        self.validation_metrics()

        #for i in range(0, self.TaskNum):
            ##task-specific index:
            #index = np.arange(0, self.args.cross_validation_fold_number)*self.TaskNum + i
            #self.Validation_Task_ROCAUC_kfoldCV.append(statistics.mean([self.Validation_Task_ROCAUC[j] for j in index]))
            #self.Validation_Task_PRAUC_kfoldCV.append(statistics.mean([self.Validation_Task_PRAUC[j] for j in index]))
            #self.Validation_Task_Loss_kfoldCV.append(statistics.mean([self.Validation_Task_Loss[j] for j in index]))
            #self.Validation_Task_Precision_kfoldCV.append(statistics.mean([self.Validation_Task_Precision[j] for j in index]))
            #self.Validation_Task_Recall_kfoldCV.append(statistics.mean([self.Validation_Task_Recall[j] for j in index]))

    def test(self):
        """
        Run inference on validation set.
        """
        if not os.path.exists(self.args.Prediction_path):
            os.makedirs(self.args.Prediction_path, exist_ok=True)
        self.train_ClassDistRatio, self.MajorityClass = label_distribution(self.train_clustered, self.args.target_name, "train")
        self.prediction_checkpoint_epoch = optimum_epoch(self.args.Metrics_path[self.fold])
        CheckPoint_path = self.args.CheckPoint_path[self.fold] + 'Model_CheckPoints_epoch' + str(self.prediction_checkpoint_epoch) + '.pt'
        try:
            checkpoint = torch.load(CheckPoint_path)
        except FileNotFoundError:
            logger.warning("Looks like the model is not trained yet. The best epoch is found to be the first epoch. "
                           "Change parameters and train again...!!!!!")
            exit()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """ Print test model's state_dict """
        # logger.info("Model's state_dict:")
        # for param_tensor in self.model.state_dict():
        #     logger.info(f"{param_tensor} \t {self.model.state_dict()[param_tensor].size()}")
        self.model.eval()
        self.test_evaluation()
        self.test_metrics_per_protein()

    def test_evaluation(self):
        """
        """
        self.validation_predictions = []
        self.validation_predictions_prob = []
        self.validation_targets = []
        self.validation_clusterCount = 0
        for cluster in self.validation_clustered.clusters:
            self.validation_clusterCount += 1
            self.do_prediction(self.validation_clustered, cluster)
        self.predictions.append(self.validation_predictions)
        self.predictions_prob.append(self.validation_predictions_prob)
        self.targets.append(self.validation_targets)

    def prepare_train_results(self, target, predictions, task_loss, cluster):
        """
        Preparing training results for calculating metrics per task.
        Results after training each cluster are concatenated to build results of one epoch for metrics calculation per epoch.
        To calculate the ROCAUC per epoch, the probability of target labels are calculated
        """
        train_nodes = self.train_clustered.sg_train_nodes[cluster]
        train_predictions = []
        train_predictions_prob = []
        for iter in range(0, self.TaskNum):
            train_predictions.append(predictions[iter][train_nodes].argmax(1).detach().cpu().clone().numpy())
            predictions_prob_0 = torch.exp(predictions[iter][train_nodes]).detach().cpu().clone().numpy()
            train_predictions_prob.append(predictions_prob_0[:, 1])
        if self.clusterCount == 1:
            self.train_predictions = torch.tensor(train_predictions).to(self.device)
            self.train_targets = target.to(self.device)
            self.train_predictions_prob = torch.tensor(train_predictions_prob).to(self.device)
        else:
            self.train_predictions = torch.cat((self.train_predictions, torch.tensor(train_predictions).to(self.device)), 1).to(self.device)
            self.train_targets = torch.cat((self.train_targets, target.to(self.device)), 1).to(self.device)
            self.train_predictions_prob = torch.cat((self.train_predictions_prob, torch.tensor(train_predictions_prob).to(self.device)), 1).to(self.device)

    def prepare_validation_results(self, validation_clustered, target, predictions, cluster):
        """
        Preparing validation results for calculating metrics per task.
        """
        validation_nodes = validation_clustered.sg_train_nodes[cluster].to(self.device)
        validation_predictions = []
        validation_predictions_prob = []
        for iter in range(0, self.TaskNum):
            validation_predictions.append(predictions[iter][validation_nodes].argmax(1).detach().cpu().clone().numpy())
            predictions_prob_0 = torch.exp(predictions[iter][validation_nodes]).detach().cpu().clone().numpy()
            validation_predictions_prob.append(predictions_prob_0[:, 1])
        if self.validation_clusterCount == 1:
            self.validation_predictions = torch.tensor(validation_predictions).to(self.device)
            self.validation_targets = target.to(self.device)
            self.validation_predictions_prob = torch.tensor(validation_predictions_prob).to(self.device)
        else:
            self.validation_predictions = torch.cat((self.validation_predictions, torch.tensor(validation_predictions).to(self.device)), 1).to(self.device)
            self.validation_targets = torch.cat((self.validation_targets, target.to(self.device)), 1).to(self.device)
            self.validation_predictions_prob = torch.cat((self.validation_predictions_prob, torch.tensor(validation_predictions_prob).to(self.device)), 1).to(self.device)

    def train_metrics(self):
        """
        Scoring the training and calculating metrics.
        :return normalized_loss: Average loss in the epoch.
        """
        warnings.filterwarnings('ignore')
        normalized_loss = self.accumulated_train_loss/self.train_node_count_seen
        normalized_task_loss = self.accumulated_train_task_loss/self.train_node_count_seen
        self.Train_Loss.append(normalized_loss)
        try:
            self.Train_BalancedAcc.append(metrics.balanced_accuracy_score(self.train_targets.cpu().reshape(-1, 1), self.train_predictions.cpu().reshape(-1, 1)))
            self.Train_Precision.append(metrics.precision_score(self.train_targets.cpu().reshape(-1,1), self.train_predictions.cpu().reshape(-1, 1), zero_division=1))
            self.Train_Recall.append(metrics.recall_score(self.train_targets.cpu().reshape(-1, 1), self.train_predictions.cpu().reshape(-1, 1), zero_division=1))
            self.Train_F1score.append(metrics.f1_score(self.train_targets.cpu().reshape(-1, 1), self.train_predictions.cpu().reshape(-1, 1), average="micro"))
            self.Train_ROCAUC.append(metrics.roc_auc_score(self.train_targets.cpu().reshape(-1, 1), self.train_predictions_prob.cpu().reshape(-1, 1)))
            precision, recall, thresholds = metrics.precision_recall_curve(self.train_targets.cpu().reshape(-1, 1), self.train_predictions_prob.cpu().reshape(-1, 1))
            self.Train_PRAUC.append(metrics.auc(recall, precision))
        except:
            logger.warning("Only one class presents for some targets. In this case metrics are not defined ...!!!!!")

        """ Metrics per tasks: """
        for i in range(0, self.TaskNum):
            try:
                self.Train_Task_Loss.append(normalized_task_loss[i])
                # self.Train_Task_Acc.append(round(self.train_predictions[i].eq(self.train_targets[i]).sum().item()/self.train_node_count_seen, 3))
                self.Train_Task_BalancedAcc.append(metrics.balanced_accuracy_score(self.train_targets.cpu()[i], self.train_predictions.cpu()[i]))
                # self.Train_Task_Precision.append(metrics.precision_score(self.train_targets.cpu()[i], self.train_predictions.cpu()[i], average="micro", zero_division=1))
                # self.Train_Task_Recall.append(metrics.recall_score(self.train_targets.cpu()[i], self.train_predictions.cpu()[i], average="micro"))
                self.Train_Task_F1score.append(metrics.f1_score(self.train_targets.cpu()[i], self.train_predictions.cpu()[i], average="micro"))
                self.Train_Task_ROCAUC.append(metrics.roc_auc_score(self.train_targets.cpu()[i], self.train_predictions_prob.cpu()[i])) # sometimes y_true has only one lable
                precision, recall, thresholds = metrics.precision_recall_curve(self.train_targets.cpu()[i], self.train_predictions_prob.cpu()[i])
                self.Train_Task_PRAUC.append(metrics.auc(recall, precision))
            except:
                logger.warning(f"Only one class presents in {self.args.target_name[i]}. In this case metrics are not defined ...!!!!!")

        """ Metrics per cluster: """
        #for cluster in self.train_clustered.clusters:
        #    for i in range(0, self.TaskNum):
        #        self.Train_Cluster_Task_Acc.append(round(self.Train_Cluster_Prediction[cluster*self.TaskNum+i].eq(self.Train_Cluster_Target[cluster*self.TaskNum+i]).sum().item()/self.train_clustering_machine.sg_train_nodes[cluster].shape[0], 3))
        #        self.Train_Cluster_Task_ROCAUC.append(metrics.roc_auc_score(self.Train_Cluster_Target[cluster*self.TaskNum+i], self.Train_Cluster_Prediction_prob[cluster*self.TaskNum+i]))


    def validation_metrics(self):
        """
        Scoring the training and calculating metrics.
        """
        warnings.filterwarnings('ignore')
        normalized_loss = self.accumulated_validation_loss/self.validation_node_count_seen
        normalized_task_loss = self.accumulated_validation_task_loss/self.validation_node_count_seen
        self.Validation_Loss.append(normalized_loss)
        try:
            #self.Validation_Acc.append(round(self.validation_predictions.reshape(-1, 1).eq(self.validation_targets.reshape(-1, 1)).sum().item()/(self.validation_node_count_seen*self.TaskNum), 3))
            self.Validation_BalancedAcc.append(metrics.balanced_accuracy_score(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions.cpu().reshape(-1, 1)))
            self.Validation_Precision.append(metrics.precision_score(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions.cpu().reshape(-1, 1), zero_division=1)) #average="micro"
            self.Validation_Recall.append(metrics.recall_score(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions.cpu().reshape(-1, 1), zero_division=1)) #average="micro", macro:This does not take label imbalance into account.
            self.Validation_F1score.append(metrics.f1_score(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions.cpu().reshape(-1, 1), average="micro", pos_label=1))
            self.Validation_ROCAUC.append(metrics.roc_auc_score(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions_prob.cpu().reshape(-1, 1)))
            precision, recall, thresholds = metrics.precision_recall_curve(self.validation_targets.cpu().reshape(-1, 1), self.validation_predictions_prob.cpu().reshape(-1, 1))
            self.Validation_PRAUC.append(metrics.auc(recall, precision))
        except:
            logger.warning("Only one class presents for some targets. In this case metrics are not defined ...!!!!!")

        """ Metrics per tasks: """
        for i in range(0, self.TaskNum):
            self.Validation_Task_Loss.append(normalized_task_loss[i])
            try:
                # self.Validation_Task_Acc.append(round(self.validation_predictions[i].eq(self.validation_targets[i]).sum().item()/self.validation_node_count_seen, 3))
                self.Validation_Task_BalancedAcc.append(metrics.balanced_accuracy_score(self.validation_targets.cpu()[i], self.validation_predictions.cpu()[i]))
                self.Validation_Task_Precision.append(metrics.precision_score(self.validation_targets.cpu()[i], self.validation_predictions.cpu()[i], average="micro", zero_division=1))
                self.Validation_Task_Recall.append(metrics.recall_score(self.validation_targets.cpu()[i], self.validation_predictions.cpu()[i], average="micro"))
                self.Validation_Task_F1score.append(metrics.f1_score(self.validation_targets.cpu()[i], self.validation_predictions.cpu()[i], average="micro"))
                self.Validation_Task_ROCAUC.append(metrics.roc_auc_score(self.validation_targets.cpu()[i], self.validation_predictions_prob.cpu()[i])) # sometimes y_true has only one lable
                precision, recall, thresholds = metrics.precision_recall_curve(self.validation_targets.cpu()[i], self.validation_predictions_prob.cpu()[i])
                self.Validation_Task_PRAUC.append(metrics.auc(recall, precision))
            except:
                logger.warning(f"Only one class presents in {self.args.target_name[i]}. In this case metrics are not defined ...!!!!!")

        """ Metrics per cluster: """
        #for cluster in self.validation_clustered.clusters:
        #    for i in range(0, self.TaskNum):
        #        self.Validation_Cluster_Task_Acc.append(round(self.Validation_Cluster_Prediction[cluster*self.TaskNum+i].eq(self.Validation_Cluster_Target[cluster*self.TaskNum+i]).sum().item()/self.validation_clustered.sg_train_nodes[cluster].shape[0], 3))
        #        self.Validation_Cluster_Task_ROCAUC.append(metrics.roc_auc_score(self.Validation_Cluster_Target[cluster*self.TaskNum+i], self.Validation_Cluster_Prediction_prob[cluster*self.TaskNum+i]))

    def test_metrics_per_protein(self):
        """
        Scoring the test results per protein.
        """
        node_ID = np.array(pd.read_csv(self.args.validation_node_proteinID_path[self.fold])["node_id"])
        protein_ID = np.array(pd.read_csv(self.args.validation_node_proteinID_path[self.fold])["protein_id_flag"])
        Protein = []
        for i in range(0, max(protein_ID)+1):
            Protein.append(node_ID[np.where(protein_ID == i)])
            self.Protein_BalancedAcc.append(metrics.balanced_accuracy_score(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions[0, Protein[i]].cpu().reshape(-1, 1)))
            self.Protein_Precision.append(metrics.precision_score(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions[0, Protein[i]].cpu().reshape(-1, 1), zero_division=1))
            self.Protein_Recall.append(metrics.recall_score(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions[0, Protein[i]].cpu().reshape(-1, 1), zero_division=1))
            self.Protein_F1score.append(metrics.f1_score(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions[0, Protein[i]].cpu().reshape(-1, 1), pos_label=1,zero_division=1))
            try:
               self.Protein_ROCAUC.append(metrics.roc_auc_score(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions_prob[0, Protein[i]].cpu().reshape(-1, 1)))
            except ValueError:
               self.Protein_ROCAUC.append(0)
            try:
               precision, recall, thresholds = metrics.precision_recall_curve(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions_prob[0, Protein[i]].cpu().reshape(-1, 1))
               self.Protein_PRAUC.append(metrics.auc(recall, precision))
            except ValueError:
               self.Protein_PRAUC.append(0)
            self.Validation_MCC.append(metrics.matthews_corrcoef(self.validation_targets[0, Protein[i]].cpu().reshape(-1, 1), self.validation_predictions[0, Protein[i]].cpu().reshape(-1, 1)))