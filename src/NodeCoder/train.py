import os
import torch
import numpy as np
import random
import time
from NodeCoder.utilities.parser import parameter_parser
from NodeCoder.graph_generator.clustering import Clustering
from NodeCoder.gcn.NodeCoder import NodeCoder_Model
from NodeCoder.gcn.NodeCoder_train import NodeCoder_Trainer
from NodeCoder.utilities.utils import tab_printer, graph_reader, feature_reader, edge_feature_reader, target_reader, DownSampling, \
  optimum_epoch, csv_writter_performance_metrics, csv_writer_prediction, plot_performance_metrics
from NodeCoder.utilities.config import logger


def main(threshold_dist:int=5, multi_task_learning:bool=False, Task:list=['y_Ligand'], centrality_feature:bool=True,
         cross_validation_fold_number:int=5, epochs=50, ):
  """
  Parsing command line parameters, generating graph data, reading saved graph data, graph decomposition,
  fitting a NodeCoder and scoring the model.
  Model parameters can be defined in parser.py
  """

  """ 
  You need to specify:
  1 - Tasks of interest like
  Task = ['y_Ligand', 'y_TRANSMEM']
  
  2 - Threshold distance in Angstrom (A) for creating graph contact network
  threshold_dist = 5
  """

  """
  Default is single-task learning unless it is specified! 
  You can train NodeCoder for different tasks separately, which is recommended. In this case, only one task is given:
  Task = ['y_Ligand'] and multi_task_learning=False.
  Or you can choose multi-task learning setup by giving more tasks as Task = ['y_Ligand', 'y_Peptide'] and setting 
  multi_task_learning=True.
  """
  args = parameter_parser(NodeCoder_usage='train', threshold_dist=threshold_dist, multi_task_learning=multi_task_learning,
                          Task=Task, centrality_feature=centrality_feature, cross_validation_fold_number=cross_validation_fold_number,
                          epochs=epochs)
  tab_printer(args)

  """ 
  Random seed initialization for reproducibility and 
  setting torch to avoid the use a nondeterministic algorithm. 
  """
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

  """ Create NodeCoder Model. """
  NodeCoder_Network = NodeCoder_Model(args)
  logger.success("NodeCoder architecture initialization done.")

  """ train on multiple folds: Cross Validation """
  for i in range(0, args.cross_validation_fold_number):
    start_time = time.time()
    logger.info(f"Clustering graphs in fold {i+1} started...")
    train_graph = graph_reader(args.train_edge_path[i])
    train_features = feature_reader(args.train_features_path[i], args.train_edge_path[i], args.centrality_feature)
    train_edge_features = edge_feature_reader(args.train_edge_feature_path[i])
    train_target = target_reader(args.train_target_path[i], args.target_name)
    if args.downSampling_majority_class == 'Yes':
      train_graph, train_features, train_edge_features, train_target = DownSampling(args, train_graph, train_features, train_edge_features, train_target)
    train_clustered = Clustering(args, args.train_protein_filename_path[i], train_graph, train_features, train_edge_features, train_target, cluster_number=args.train_cluster_number)
    train_clustered.decompose()
    logger.info(f"Clustering train graph completed in {(time.time() - start_time)} seconds.")

    start_time = time.time()
    validation_graph = graph_reader(args.validation_edge_path[i])
    validation_edge_features = edge_feature_reader(args.validation_edge_feature_path[i])
    validation_features = feature_reader(args.validation_features_path[i], args.validation_edge_path[i], args.centrality_feature)
    validation_target = target_reader(args.validation_target_path[i], args.target_name)
    validation_clustered = Clustering(args, args.validation_protein_filename_path[i], validation_graph, validation_features, validation_edge_features, validation_target, cluster_number=args.validation_cluster_number)
    validation_clustered.decompose()
    logger.info(f"Clustering validation graph completed in {(time.time() - start_time)} seconds.")

    logger.info(f"Training NodeCoder on fold {i+1} started ...")
    trainer = NodeCoder_Trainer(args, NodeCoder_Network.model, train_clustered, validation_clustered, i)
    trainer.train()
    logger.success(f"Training NodeCoder on fold {i+1} completed.")
    logger.info("Performance metrics are being saved to disk ...")
    csv_writter_performance_metrics(trainer, i)

    """ 
    Now we find the optimum epoch and load the optimum trained model that is saved using checkpoints.
    Then run inference to regenerate the final predicted labels and calculate prediction scores per protein.
    """
    logger.info(f"Running inference on validation fold {i+1} with model saved at optimum epoch ...")
    checkpoint_epoch = optimum_epoch(args.Metrics_path[i])
    inference = NodeCoder_Trainer(args, NodeCoder_Network.model, train_clustered, validation_clustered, i, checkpoint_epoch)
    inference.test()
    logger.success(f"Inference for fold {i+1} completed.")
    logger.info("Calculating and writing prediction scores per protein ...")
    csv_writer_prediction(args.NodeCoder_usage, Task, inference.validation_targets, inference.validation_predictions, inference.validation_predictions_prob, args.validation_node_proteinID_path[i], args.Prediction_fileName[i])
    logger.success(f"Model is successfully trained on fold {i+1} and prediction scores are saved!")

  plot_performance_metrics(args)

if __name__ == "__main__":
  main()