import os
import torch
import numpy as np
import random
import time
from parser import parameter_parser
from graph_data_generator import Graph_Data_Generator
from clustering import Clustering
from NodeCoder import NodeCoder_Model
from NodeCoder_train import NodeCoder_Trainer
from utils import colors, tab_printer, graph_reader, feature_reader, edge_feature_reader, target_reader, DownSampling, \
  optimum_epoch, csv_writter_performance_metrics, csv_writer_prediction, plot_performance_metrics

def main():
  """
  Parsing command line parameters, generating graph data, reading saved graph data, graph decomposition,
  fitting a NodeCoder and scoring the model.
  Model parameters can be defined in parser.py
  """

  """ 
  Here you need to specify:
  Tasks of interest
  Threshold distance for creating graph contact network
  """
  Task = ['y_Ligand']
  threshold_dist = 5 #(A)

  """ default is single-task learning unless it is specified! """
  args = parameter_parser(Task, NodeCoder_usage='train', threshold_dist=threshold_dist, multi_task_learning='No')
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

  """ 
  Generate separate protein graphs for train and validation.
  This class save generated graph data and avoid regenerating them.
  """
  start_time = time.time()
  Graph_Data = Graph_Data_Generator(args)
  Graph_Data.graph_data_files_generator()
  print("--- %s seconds for generating graph data files. ---" %(time.time() - start_time))

  """ Create NodeCoder Model. """
  NodeCoder_Network = NodeCoder_Model(args)

  """ train on multiple folds: Cross Validation """
  for i in range(0, args.cross_validation_fold_number):
    start_time = time.time()
    print(colors.HEADER + "\n--- clustering graphs in fold %s started ..." %(i+1) + colors.ENDC)
    train_graph = graph_reader(args.train_edge_path[i])
    train_features = feature_reader(args.train_features_path[i], args.train_edge_path[i])
    train_edge_features = edge_feature_reader(args.train_edge_feature_path[i])
    train_target = target_reader(args.train_target_path[i], args.target_name)
    if args.downSampling_majority_class == 'Yes':
      train_graph, train_features, train_edge_features, train_target = DownSampling(args, train_graph, train_features, train_edge_features, train_target)
    train_clustered = Clustering(args, args.train_protein_filename_path[i], train_graph, train_features, train_edge_features, train_target, cluster_number=args.train_cluster_number)
    train_clustered.decompose()
    print("--- %s seconds for clustering train graph ---" %((time.time() - start_time)))

    start_time = time.time()
    validation_graph = graph_reader(args.validation_edge_path[i])
    validation_edge_features = edge_feature_reader(args.validation_edge_feature_path[i])
    validation_features = feature_reader(args.validation_features_path[i], args.validation_edge_path[i])
    validation_target = target_reader(args.validation_target_path[i], args.target_name)
    validation_clustered = Clustering(args, args.validation_protein_filename_path[i], validation_graph, validation_features, validation_edge_features, validation_target, cluster_number=args.validation_cluster_number)
    validation_clustered.decompose()
    print("--- %s seconds for clustering validation graph ---" %((time.time() - start_time)))

    print(colors.HEADER + "\n--- training NodeCoder on fold %s started ... " %(i+1) + colors.ENDC)
    trainer = NodeCoder_Trainer(args, NodeCoder_Network.model, train_clustered, validation_clustered, i)
    trainer.train()
    print(colors.HEADER + "\n--- training NodeCoder on fold %s completed. ---" %(i+1) + colors.ENDC)
    print(colors.HEADER + "\n--- Performance metrics are being saved to disk. ---" + colors.ENDC)
    csv_writter_performance_metrics(trainer, i)

    """ 
    Now we find the optimum epoch and load the optimum trained model that is saved using checkpoints.
    Then run inference to regenerate the final predicted labels and calculate prediction scores per protein.
    """
    print(colors.HEADER + "\n--- Running inference on validation fold %s with model saved at optimum epoch ... " %(i+1) + colors.ENDC)
    checkpoint_epoch = optimum_epoch(args.Metrics_path[i])
    inference = NodeCoder_Trainer(args, NodeCoder_Network.model, train_clustered, validation_clustered, i, checkpoint_epoch)
    inference.test()
    print(colors.HEADER + "\n--- Inference completed. ---" + colors.ENDC)
    print(colors.HEADER + "\n--- Calculating and writing prediction scores per protein ... " + colors.ENDC)
    csv_writer_prediction(Task, inference.validation_targets, inference.validation_predictions, inference.validation_predictions_prob, args.validation_node_proteinID_path[i], args.Prediction_fileName[i])

  plot_performance_metrics(args)

if __name__ == "__main__":
  main()
