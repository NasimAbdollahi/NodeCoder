import numpy as np
import pandas as pd
import torch
import random
import networkx as nx
from texttable import Texttable
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from NodeCoder.utilities.config import logger


class colors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

def tab_printer(args):
  """
  Function to print the logs in a nice tabular format.
  :param args: Parameters used for the model.
  """
  args = vars(args)
  keys = sorted(args.keys())
  table = Texttable()
  table.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
  print(colors.OKGREEN + table.draw() + colors.ENDC)

def csv_files_SanityCheck(Graph,name, path):
  DataFrame = pd.DataFrame(list(zip(Graph.protein_files_name, Graph.node_num, Graph.protein_frame_Nan_Count)), columns=['Protein File', 'Node Num', 'Removed NaNs'])
  DataFrame.to_csv(path + name + '_ProteinFiles.csv', index=False)

def csv_writter_known_proteins(protein_files, node_num, path, filename):
  task_files = protein_files
  features_files = [protein_files[iter].replace('tasks', 'features') for iter in range(0, len(protein_files))]
  DataFrame = pd.DataFrame(list(zip(task_files, features_files, node_num)), columns=['tasks file', 'features file', 'node num'])
  DataFrame.to_csv(path + filename, index=False)

def csv_writter_grouping_protein(path, name, protein_filenames):
  task_files = protein_filenames
  features_files = [protein_filenames[iter].replace('tasks', 'features') for iter in range(0, len(protein_filenames))]
  DataFrame = pd.DataFrame(list(zip(task_files, features_files)), columns=['tasks file', 'features file'])
  DataFrame.to_csv(path + name + '_ProteinFileNames.csv', index=False)

def csv_writter_graph_data(Graph,name,target_output, path):
  """ Write list of protein files and their node numbers """
  protein_files_DataFrame = pd.DataFrame(list(zip(Graph.protein_files_name, Graph.node_num, Graph.protein_Nan_Count)), columns=['Protein File', 'Node Num', 'Removed NaNs'])
  protein_files_DataFrame.to_csv(path + name + '_ProteinFiles.csv', index=False)

  """ Write edge files """
  edge_files_DataFrame = pd.DataFrame(list(zip(Graph.edge_node1, Graph.edge_node2)), columns=['id1', 'id2'])
  edge_files_DataFrame.to_csv(path + name + '_edges.csv', index=False)

  """ Write features files """
  node_id = np.array([i*np.ones(len(Graph.node_features[1]), dtype=int) for i in range(len(Graph.node_features))]).reshape(-1)
  feature_id = np.tile([i for i in range(len(Graph.node_features[1]))], len(Graph.node_features))
  node_features = np.array(Graph.node_features).reshape(-1)
  features_files_DataFrame = pd.DataFrame(list(zip(node_id, feature_id, node_features)), columns=['node_id', 'feature_id', 'value'])
  features_files_DataFrame.to_csv(path + name + '_features.csv', index=False)

  """ Write target files """
  Target = np.zeros((len(Graph.node_features), len(Graph.labels)), dtype=int)
  for l in range(0, len(Graph.labels)):
    Target[:, l] = np.array(Graph.labels[l], dtype=int)
  target_frame = pd.DataFrame(list(Target), columns=target_output)
  target_frame.to_csv(path + name + '_target.csv', index=False)

  """ Write edge features files """
  edge_features_1 = [Graph.edge_features[i][0] for i in range(0, len(Graph.edge_features))]
  edge_features_2 = [Graph.edge_features[i][1] for i in range(0, len(Graph.edge_features))]
  edge_features_3 = [Graph.edge_features[i][2] for i in range(0, len(Graph.edge_features))]
  edge_features_files_DataFrame = pd.DataFrame(list(zip(Graph.edge_node1, Graph.edge_node2, edge_features_1, edge_features_2, edge_features_3)),
                           columns=['id1', 'id2', 'edge_length', 'edge_cosine_angle', 'edge_sequence_distance'])
  edge_features_files_DataFrame.to_csv(path + name + '_edge_features.csv', index=False)

  """ Write nodes and ProteinID files """
  protein_id = [item for sublist in Graph.nodes_protein_id for item in sublist]
  protein_id_flag = [item for sublist in Graph.nodes_protein_id_flag for item in sublist]
  node_id = [i for i in range(0, len(protein_id))]
  nodes_ProteinID_files_DataFrame = pd.DataFrame(list(zip(node_id, protein_id_flag, protein_id)), columns=['node_id', 'protein_id_flag', 'protein_id'])
  nodes_ProteinID_files_DataFrame.to_csv(path + name + '_nodes_ProteinID.csv', index=False)

def graph_reader(path):
  """
  Function to read the graph from the path.
  :param path: Path to the edge list.
  :return graph: NetworkX object returned.
  """
  try:
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
  except FileNotFoundError:
    logger.critical("file containing graph data (contact network) could not be found.")
    logger.info("graph files are required to be generated before attempting to train NodeCoder.")
    raise
  return graph

def feature_reader(path, edge_path: str='NA', centrality_feature:bool=False):
  """
  Reading the sparse feature matrix stored as csv from the disk.
  :param path: Path to the csv file.
  :return features: Dense matrix of features.
  """
  try:
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = np.array(node_index).max()+1
    feature_count = np.array(feature_index).max()+1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    """ Add Node Centrality Feature (Node Degree or Weighted Node Degree: average of edge features) """
    if centrality_feature:
      edges = pd.read_csv(edge_path)
      feature_centrality = np.array(edges['id1'].value_counts().sort_index()).reshape(features.shape[0], 1)
      features = np.concatenate((features, feature_centrality), axis=1)
  except FileNotFoundError:
    logger.critical("file containing node features could not be found.")
    logger.info("graph files are required to be generated before attempting to train NodeCoder.")
    raise
  except Exception as e:
    logger.error(e)
  return features

def feature_reader_leave_one_feature(path, f):
  """
  Reading the sparse feature matrix stored as csv from the disk.
  :param path: Path to the csv file.
  :return features: Dense matrix of features.
  Leaves feature "f" out
  """
  try:
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = np.array(node_index).max()+1
    feature_count = np.array(feature_index).max()+1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    features = np.delete(features, f, axis=1)
  except:
    logger.critical("file containing node features could not be found.")
    logger.info("graph files are required to be generated before attempting to train NodeCoder.")
    raise
  return features

def edge_feature_reader(path):
  """
  Reading the edge features stored as csv from the disk.
  :param path: Path to the csv file.
  :return features: edge features vector.
  """
  try:
    edge_features = np.array(pd.read_csv(path))
  except:
    logger.critical("file containing edge features could not be found.")
    logger.info("graph files are required to be generated before attempting to train NodeCoder.")
    raise
  return edge_features

def target_reader(path, target_name):
  """
  Reading the target vector from disk.
  :param path: Path to the target.
  :return target: Target vector.
  includes all tasks in target list or only some chosen targets!
  """
  try:
    target = []
    for i in target_name:
      target.append(np.array(pd.read_csv(path)[i]).tolist())
    target = np.array(target).transpose()
  except:
    logger.critical("file containing target labels could not be found.")
    logger.info("graph files are required to be generated before attempting to train NodeCoder.")
    raise
  return target

def ProteinID_reader(path):
  ProteinID = pd.read_csv(path)
  for i in range(0, len(ProteinID['protein_id'])):
    ProteinID.loc[i, 'protein_id'] = ProteinID.loc[i, 'protein_id'].split(".", -1)[0]
  return ProteinID

def optimum_epoch(path):
  epoch = np.array(pd.read_csv(path)['Epoch id']).tolist()
  ROCAUC = np.array(pd.read_csv(path)['Validation ROC_AUC']).tolist()
  #PRAUC = np.array(pd.read_csv(path)['Validation PR_AUC']).tolist()
  best_epoch = epoch[ROCAUC.index(max(ROCAUC))]
  if best_epoch == 0:
    logger.warning("NodeCoder is not learning: best epoch is found to be the first epoch. You need to increase training "
                   "epoch or change model parameters...!!!!!")
    exit()
  return best_epoch

def Positive_Expansion(graph, target):
  PositiveNodes = []
  for i in range(0, np.shape(target)[1]):
    nodes = np.nonzero(target[:, i]) #Positive nodes
    PositiveNodes.append(nodes)
  """ finding immediate neighbours: """
  for i in range(0, len(PositiveNodes)):
    nodes = PositiveNodes[i]
    for j in range(0, np.shape(nodes)[1]):
      immediate_neighbours = graph.edges(nodes[0][j])
      for m, n in enumerate(immediate_neighbours):
        target[n[1], i] = 1
  return target

def DownSampling(args, graph, features, edge_features, target):
  """ the current function can only be applied for single-task learning scenario """
  np.random.seed(args.seed)
  random.seed(args.seed)
  if args.downSampling_majority_class == 'Yes':
    final_positive_label_ratio = 1 - args.downSampling_majority_class_ratio

    """ choose all positive node: """
    positive_nodes = [i for i in np.array(list(graph.nodes)) if target[i] == 1]
    """ random negative node selection: """
    negative_nodes = [i for i in np.array(list(graph.nodes)) if target[i] == 0]
    negative_nodes_num = round(len(positive_nodes)/final_positive_label_ratio - len(positive_nodes))
    selected_negative_nodes = np.random.choice(negative_nodes, size=(negative_nodes_num,), replace=False).tolist()
    selected_nodes = sorted(np.array(positive_nodes + selected_negative_nodes))
    not_selected_nodes = list(set(list(graph.nodes))^set(selected_nodes))
    if (len(set(not_selected_nodes)) == len(not_selected_nodes)):
      logger.info("DownSampling: all elements are unique.")
    else:
      logger.info("DownSampling: elements are not unique.")

    """ updating the graph: removes node and all edges connected to the node: """
    for i in not_selected_nodes:
      graph.remove_node(i)
    """ 
    edgefeatures - the index of selected edges are extracted in order to be used 
    for updating the edge features 
    """
    if args.includeEdgeFeature == 'Yes':
      selected_edges = list(graph.edges)
      all_edge = [(int(edge_features[i, 0]), int(edge_features[i, 1])) for i in range(0, edge_features.shape[0])]
      edge_index = [all_edge.index(selected_edges[i]) for i in range(0, len(selected_edges))]
    """ 
    Relabeling: after updating the graph, node index is non contiguous. 
    update node idx and update respective edges 
    """
    mapping = dict(zip(sorted(graph), range(0, len(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    """ updating the edge features - Optimized: """
    if args.includeEdgeFeature == 'Yes':
      edge_features = edge_features[edge_index[:], :]
    else:
      edge_features = 'NA'
    """ updating the target and features: """
    target = target[selected_nodes]
    features = features[selected_nodes]
  return graph, features, edge_features, target

def cluster_membership_reader(path):
  """
  Reading the target vector from disk.
  :param path: Path to the cluster_membership.
  :return target: cluster_membership data.
  """
  cluster_mem = np.array(pd.read_csv(path))
  cluster_membership = dict(enumerate(cluster_mem[:, 1]))
  return cluster_membership

def protein_clustering(protein_filename,Proteins_Node, cluster_Num):
  """
  To include all protein files in clusters, Res_File = PNum % cluster_Num files are included in
  the later clusters by adding one more file to each
  """
  PNum = protein_filename.shape[0]
  Res_File = PNum % cluster_Num
  """ the node number for protein files in each cluster: """
  Protein_per_Cluster_1 = PNum//cluster_Num
  Protein_per_Cluster_2 = Protein_per_Cluster_1 + 1
  cluster_Num1 = cluster_Num - Res_File
  num_cluster = 1
  node_num_per_cluster = []
  node_protein_per_cluster = []
  node_num = 0
  protein_count = 0
  for i in range(0, len(protein_filename)):
    node_num += Proteins_Node[i]
    protein_count += 1
    """ get node num for protein files in each cluster: """
    if num_cluster <= cluster_Num1:
      if protein_count == Protein_per_Cluster_1:
        node_num_per_cluster.append(node_num)
        node_protein_per_cluster.append(protein_count)
        protein_count = 0
        node_num = 0
        num_cluster += 1
    else:
      if protein_count == Protein_per_Cluster_2:
        node_num_per_cluster.append(node_num)
        node_protein_per_cluster.append(protein_count)
        protein_count = 0
        node_num = 0
        num_cluster += 1
  Cluster = [i for i in range(cluster_Num)]
  Cluster_Cluster = np.array(Cluster[0]*np.ones(node_num_per_cluster[0])).astype(np.int32)
  for nc in range(1, cluster_Num):
    Cluster_Cluster = np.concatenate((Cluster_Cluster, np.array(Cluster[nc]*np.ones(node_num_per_cluster[nc])).astype(np.int32)), axis=0)
  cluster_membership = dict(enumerate(Cluster_Cluster))
  return cluster_membership

def csv_writter_performance_metrics(Results, i):
  """ Writing the model performance metrics to disk. """
  Columns = ['Epoch id', 'Train Loss', 'Train ROC_AUC', 'Train PR_AUC', 'Train_Precision', 'Train_Recall',
             'Validation Loss', 'Validation ROC_AUC', 'Validation PR_AUC', 'Validation_Precision', 'Validation_Recall']
  performance_DataFrame = pd.DataFrame(list(zip(Results.Performance_epochs, Results.Train_Loss,Results.Train_ROCAUC, Results.Train_PRAUC,
                                               Results.Train_Precision, Results.Train_Recall, Results.Validation_Loss,
                                               Results.Validation_ROCAUC, Results.Validation_PRAUC, Results.Validation_Precision,
                                               Results.Validation_Recall)), columns=Columns)
  performance_DataFrame.to_csv(Results.args.Metrics_path[i], index=False)

  # """ metrics per tasks """
  # Columns = ['Task id', 'Epoch id', 'Train Loss', 'Train ROC_AUC', 'Train PR_AUC', 'Validation Loss', 'Validation ROC_AUC',
  #            'Validation PR_AUC']
  # Task_id = np.tile([i for i in range(Results.TaskNum)], len(Epoch_id))
  # Epoch_id_2 = np.array([i*np.ones(Results.TaskNum, dtype=int) for i in Epoch_id]).reshape(-1)
  # task_performance_DataFrame = pd.DataFrame(list(zip(Task_id, Epoch_id_2, Results.Train_Task_Loss, Results.Train_Task_ROCAUC,
  #                                                    Results.Train_Task_PRAUC, Results.Validation_Task_Loss_kfoldCV,
  #                                                    Results.Validation_Task_ROCAUC_kfoldCV, Results.Validation_Task_PRAUC_kfoldCV)), columns=Columns)
  # task_performance_DataFrame.to_csv(Results.args.Metrics_task_path[i], index=False)

def csv_writer_performance_metrics_perprotein(Results, i):
  """ Writing per protein prediction score to disk. """
  protein_id_DataFrame = pd.read_csv(Results.args.validation_node_proteinID_path[i])
  protein_DataFrame = protein_id_DataFrame.drop_duplicates(subset=["protein_id"], keep="last", ignore_index=True)
  del protein_DataFrame['node_id']
  prediction_score_DataFrame = pd.DataFrame(list(zip(Results.Protein_ROCAUC, Results.Protein_PRAUC, Results.Protein_BalancedAcc,
                                                     Results.Protein_F1score, Results.Protein_MCC)), columns=['ROCAUC', 'PRAUC', 'BalancedACC', 'F1Score', 'MCC'])
  final_DataFrame = pd.concat([protein_DataFrame, prediction_score_DataFrame], axis=1)
  final_DataFrame.to_csv(Results.args.Prediction_Metrics_filename[i], index=False)

def csv_writer_prediction(NodeCoder_usage, tasks, target, predictions, predictions_prob, proteinID_path, path_to_save):
  """
  Writing the final prediction to disk, csv file.
  """
  tasks = [t.split('_') for t in tasks]
  Tasks = []
  for t in tasks:
    if len(t) == 2:
      Tasks.append(t[1])
    else:
      Tasks.append('_'.join(t[1:]))

  firstrow = []
  firstrow.extend([t+' Target', t+' Prediction', t+' Prediction Probability'] for t in Tasks)
  Columns = [j for i in firstrow for j in i]

  if NodeCoder_usage == 'train':
    if len(Tasks) == 1:
      Rows = [np.array(target[0].cpu()).squeeze(), np.array(predictions[0].cpu()).squeeze(), np.array(predictions_prob[0].cpu()).squeeze()]
    else:
      Rows0 = [[np.array(target[i].cpu()).squeeze(), np.array(predictions[i].cpu()).squeeze(), np.array(predictions_prob[i].cpu()).squeeze()] for i in range(0, len(Tasks))]
      Rows = [j for i in Rows0 for j in i]
  elif NodeCoder_usage == 'predict':
    if len(Tasks) == 1:
      Rows = [np.array(target[0][0].cpu()).squeeze(), np.array(predictions[0][0].cpu()).squeeze(), np.array(predictions_prob[0][0].cpu()).squeeze()]
    else:
      Rows0 = [[np.array(target[i][0].cpu()).squeeze(), np.array(predictions[i][0].cpu()).squeeze(), np.array(predictions_prob[i][0].cpu()).squeeze()] for i in range(0, len(Tasks))]
      Rows = [j for i in Rows0 for j in i]

  """ To include the protein ID  """
  prediction_DataFrame = pd.read_csv(proteinID_path)
  for i in range(0, len(Columns)):
    DF = pd.DataFrame({Columns[i]: Rows[i]})
    prediction_DataFrame = pd.concat([prediction_DataFrame, DF], axis=1)
  prediction_DataFrame.to_csv(path_to_save, index=False)

def label_distribution(Data, Tasks, title):
  train_ClassDistRatio = []
  MajorityClass = []
  label_dist = []
  clusterCount = 0
  for cluster in Data.clusters:
    clusterCount += 1
    if clusterCount == 1:
      nodes = Data.sg_train_nodes[cluster].clone().detach()
    else:
      nodes = torch.cat((nodes, Data.sg_train_nodes[cluster]), 0)

  target = np.array(Data.target)
  for i in range(0, len(Tasks)):
    all_nodes = np.array(nodes)
    label_dist.append(np.count_nonzero(target[all_nodes, i])*100/all_nodes.shape[0])
    """ Label Distribution in Clusters: """
    label_dist_cluster = np.zeros(len(Data.clusters))
    for cluster in Data.clusters:
      nodes_cluster = Data.sg_train_nodes[cluster].clone().detach()
      label_dist_cluster[cluster] = np.count_nonzero(target[nodes_cluster, i])*100/nodes_cluster.shape[0]

    """ the ratio of the majority class to minority class to calculate weights for loss: """
    try:
      if np.count_nonzero(target[nodes, i]) > (len(nodes) - np.count_nonzero(target[nodes, i])):
        train_ClassDistRatio.append(np.count_nonzero(target[nodes, i])/(len(nodes) - np.count_nonzero(target[nodes, i])))
        MajorityClass.append(1)
      else:
        train_ClassDistRatio.append((len(nodes) - np.count_nonzero(target[nodes, i]))/np.count_nonzero(target[nodes, i]))
        MajorityClass.append(0)
      logger.info(f"{label_dist[i]} percent positive labels for {Tasks[i]} in {title} data.")
      logger.info(f"distribution in clusters:{label_dist_cluster}")
    except:
      logger.critical(f"There is no positive label in {title} data for {Tasks[i]} ")
      MajorityClass.append(0)

  return train_ClassDistRatio, MajorityClass

def plot_metrics(file_name, title, Scheme, zoomOption, PerformanceStep):
  """ plots train and validation metrics on separate figures. """
  colors = ['#CC79A7', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#000000', 'pink', 'lightblue']
  fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 7))
  for i in range(0, len(file_name)):
    epoch = np.array(pd.read_csv(file_name[i])["Epoch id"]) + 1
    metric1 = np.array(pd.read_csv(file_name[i])["Train ROC_AUC"])
    metric2 = np.array(pd.read_csv(file_name[i])["Train PR_AUC"])
    metric3 = np.array(pd.read_csv(file_name[i])["Train Loss"])
    metric4 = np.array(pd.read_csv(file_name[i])["Validation ROC_AUC"])
    metric5 = np.array(pd.read_csv(file_name[i])["Validation PR_AUC"])
    metric6 = np.array(pd.read_csv(file_name[i])["Validation Loss"])

    if PerformanceStep == 'Yes':
      epoch = np.array([epoch[n] for n in range(epoch.shape[0]) if n % 2 == 0])
      metric1 = np.array([metric1[n] for n in range(metric1.shape[0]) if n % 2 == 0])
      metric2 = np.array([metric2[n] for n in range(metric2.shape[0]) if n % 2 == 0])
      metric3 = np.array([metric3[n] for n in range(metric3.shape[0]) if n % 2 == 0])
      metric4 = np.array([metric4[n] for n in range(metric4.shape[0]) if n % 2 == 0])
      metric5 = np.array([metric5[n] for n in range(metric5.shape[0]) if n % 2 == 0])
      metric6 = np.array([metric6[n] for n in range(metric6.shape[0]) if n % 2 == 0])

    if zoomOption == 'No':
      axs[0, 0].plot(epoch, metric1, color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[0, 1].plot(epoch, metric2, color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[0, 2].plot(epoch, metric3, color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 0].plot(epoch, metric4, color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 1].plot(epoch, metric5, color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 2].plot(epoch, metric6, color=colors[i], linewidth=2, label='%s'%Scheme[i])
    else:
      zoomNum = 4
      endNum = len(epoch)
      axs[0, 0].plot(epoch[zoomNum:endNum], metric1[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[0, 1].plot(epoch[zoomNum:endNum], metric2[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[0, 2].plot(epoch[zoomNum:endNum], metric3[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 0].plot(epoch[zoomNum:endNum], metric4[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 1].plot(epoch[zoomNum:endNum], metric5[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])
      axs[1, 2].plot(epoch[zoomNum:endNum], metric6[zoomNum:endNum], color=colors[i], linewidth=2, label='%s'%Scheme[i])

  axs[0, 0].set_title('Train ROCAUC', fontweight="bold")
  axs[0, 1].set_title('Train PRAUC', fontweight="bold")
  axs[0, 2].set_title('Train Loss', fontweight="bold")
  axs[1, 0].set_title('Validation ROCAUC', fontweight="bold")
  axs[1, 1].set_title('Validation PRAUC', fontweight="bold")
  axs[1, 2].set_title('Validation Loss', fontweight="bold")
  """ adding horizontal and vertical grid lines """
  for i in range(0, 2):
    for j in range(0, 3):
      axs[i, j].yaxis.grid(True)
      axs[i, j].xaxis.grid(True)
      if i == 1:
        axs[i, j].set_xlabel('epoch', fontweight="bold")
      axs[i, j].legend()
  plt.suptitle(title, fontsize=16, fontweight="bold")
  plt.show()
  image_name = file_name[0].split('Metrics_Fold')[0] + 'Curves.jpg'
  plt.savefig(image_name)

def plot_performance_metrics(args):
  """ plot name: """
  if len(args.target_name) == 1:
    t0 = args.target_name[0].split('_')
    if len(t0) > 2:
      target_name = t0[1] + '_' + t0[2]
    else:
      target_name = t0[1]

  else:
    target_name = [t.split('_')[-1] for t in args.target_name]
    target_name = ' & '.join(target_name)
    target_name = 'Multitask Learning: ' + target_name

  graph = str(args.threshold_dist)+'A'
  folds = ['fold ' + str(L) for L in range(1, args.cross_validation_fold_number+1)]
  PerformanceStep = 'Yes'
  zoomOption = 'No'
  plot_name = target_name + ' - ' + graph + ' Graph'
  plot_metrics(args.Metrics_path, plot_name, folds, zoomOption, PerformanceStep)



