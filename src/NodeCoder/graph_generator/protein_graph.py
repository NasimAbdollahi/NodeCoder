import numpy as np
import pandas as pd


class protein_graph_generator(object):
  def __init__(self, path_featurized_data, protein_tasks_files: list, protein_features_files: list, target_output: list,
               threshold_distance: float = 5):
    """
    Class for generating graphs ready to be used for GCN modeling
    :param protein_tasks_files: list of proteins tasks files
    :param protein_features_files: list of proteins features files
    :param target_output: output columns to be used, in multi-task learning setting, for label list generation
    :param distance_threshold: distance threshold for edge generation based on cartesian coordinates
    """
    self.path_data = path_featurized_data
    self.protein_tasks_files = protein_tasks_files
    self.protein_features_files = protein_features_files
    self.threshold_distance = threshold_distance
    self.target_output = target_output

    self.node_features = []
    self.labels = []
    self.edge_node1 = []
    self.edge_node2 = []
    self.edge_features = []
    self.edge_length = []
    self.edge_cosine_dist = []
    self.seq_dist = []

    self.protein_tasks_files_name = []
    self.protein_features_files_name = []
    self.protein_files_name = []
    self.nodes_protein_id = []
    self.nodes_protein_id_flag = []
    self.node_num = []
    self.protein_Nan_Count = []
    self.residue_index = 0

    self.labels = [[] for iter in range(0, len(self.target_output))]


  def data_prep(self):
    """
    extrating information from protein files
    :return: dataframes of coordiantes, protein feaures, and output values
    This module considers secondary structurs (y_dssp)s as features if self.nodefeature_selection == 'Include dssp'!
    """
    coordinate_frame = self.protein_frame.filter(regex='coord')
    feature_frame1 = self.protein_frame.filter(regex='feat').astype(float)

    """ Distance from center of protein """
    ProteinCenter = [coordinate_frame['coord_X'].mean(), coordinate_frame['coord_Y'].mean(), coordinate_frame['coord_Z'].mean()]
    coordinates = list(coordinate_frame.iloc[:, :].values)
    distance_fromCenter = [self.dist(coords_1=coordinates[nodei], coords_2=ProteinCenter) for nodei in range(0, coordinate_frame.shape[0])]
    distance_fromCenter_normalized = [(distance_fromCenter[nodei]-min(distance_fromCenter))/(max(distance_fromCenter)-min(distance_fromCenter)) for nodei in range(0, coordinate_frame.shape[0])]
    feature_frame2 = pd.DataFrame({'feat_CentricDist':distance_fromCenter_normalized})
    feature_frame2.set_index(coordinate_frame.index, inplace=True)

    """ Cosine Distance with respect to center of protein """
    cosine_dist_fromCenter = [self.CosineDist(coords_1=coordinates[nodei], coords_2=np.array(ProteinCenter)) for nodei in range(0, coordinate_frame.shape[0])]
    feature_frame3 = pd.DataFrame({'feat_CentricCosineDist':cosine_dist_fromCenter})
    feature_frame3.set_index(coordinate_frame.index, inplace=True)

    """ 
    i+ & i- residue sequence info:
    by assigning values 1-20 to sequence then normalized to 0-1 (reserving 0 for first and last residues 
    where they do not have i- and i+ respectively) 
    """
    Sequence_feat_id = ['feat_' + L for L in 'ACDEFGHIKLMNPQRSTVWY']
    Sequence_data = self.protein_frame.filter(items=Sequence_feat_id)
    a = np.column_stack(np.where(np.array(Sequence_data)))[:, 1]
    """ reserve zero for first and last residues: """
    a += 1
    normalized_sequence_info0 = (a-a.min())/(a.max()-a.min())
    iPlus = []
    iMinus = []
    for j in range(0, len(Sequence_data)):
      if j == 0:
        iMinus.append(0)
        iPlus.append(normalized_sequence_info0[j+1])
      elif j == len(Sequence_data)-1:
        iMinus.append(normalized_sequence_info0[j-1])
        iPlus.append(0)
      else:
        iMinus.append(normalized_sequence_info0[j-1])
        iPlus.append(normalized_sequence_info0[j+1])
    feature_frame4 = pd.DataFrame(list(zip(iPlus, iMinus)), columns=['feat_iPlus', 'feat_iMinus'])
    feature_frame4.set_index(coordinate_frame.index, inplace=True)

    feature_frame = pd.concat([feature_frame1, feature_frame2, feature_frame3, feature_frame4], axis=1)
    sequence_data = self.protein_frame.loc[:, 'annotation_sequence'].index.tolist()
    if len(self.target_output) > 0:
      output_frame = self.protein_frame[self.target_output]
    else:
      output_frame = self.protein_frame.filter(regex='y_').astype(int)

    return coordinate_frame, feature_frame, output_frame, sequence_data

  # def get_nodenum(self):
  #     node_num = 0
  #     for file_iter in self.protein_files:
  #         with gzip.open(file_iter, 'rb') as f:
  #             protein_frame = pickle.load(f)
  #         node_num = node_num+protein_frame.shape[0]
  #     self.node_num  = node_num

  def dist(self, coords_1: list, coords_2: list):
    """
    calculating distance between two nodes
    :param coords_1: Cartesian coordiantes of nodes one
    :param coords_2: Cartesian coordiantes of nodes two
    :return: distance between the two
    """
    return np.sqrt(sum((coords_1-coords_2)**2))

  def CosineDist(self, coords_1: list, coords_2: list):
    """
    calculating angle between two nodes
    :param coords_1: Cartesian coordiantes of nodes one
    :param coords_2: Cartesian coordiantes of nodes two
    :return: cosine distance between the two
    """
    cosine_dist = np.dot(coords_1, coords_2)/(np.sqrt(sum((coords_1)**2))*np.sqrt(sum((coords_2)**2)))
    return cosine_dist

  def graph_gen(self, coordinate_frame: pd.DataFrame,
                feature_frame: pd.DataFrame,
                output_frame: pd.DataFrame, sequence_data):
    """
    Generating lists necessary to build a GCN model including node features, labels list,
    :param coordinate_frame: dataframe of x, y, z coordinates of nodes
    :param feature_frame: dataframe of features of nodes
    :param output_frame: dataframe of output values of nodes
    """

    """ [node1_featurelist, node2_featurelist, node3_featurelist, ...]: """
    self.node_features = self.node_features + feature_frame.values.tolist()
    output_list = output_frame.transpose().values.tolist()
    """ [list of labels for protein 1, list of labels for protein 2]: """
    self.labels = [self.labels[out_iter] + output_list[out_iter] for out_iter in range(0, len(output_list))]
    coordinates = list(coordinate_frame.iloc[:, :].values)

    for node1_iter in range(0, coordinate_frame.shape[0]):
      node1_index = node1_iter + self.residue_index
      node1_coords = coordinates[node1_iter]
      for node2_iter in range(0, coordinate_frame.shape[0]):
        node2_coords = coordinates[node2_iter]
        distance = self.dist(coords_1=node1_coords, coords_2=node2_coords)
        edge_cosine_dist = self.CosineDist(coords_1=node1_coords, coords_2=node2_coords)
        seq_distance = sequence_data[node1_iter] - sequence_data[node2_iter]

        if distance > 0 and distance <= self.threshold_distance:
          self.edge_node1.append(node1_index)
          self.edge_node2.append(node2_iter + self.residue_index)
          self.edge_length.append(distance)
          self.edge_cosine_dist.append(edge_cosine_dist)
          self.seq_dist.append(seq_distance)
          self.edge_features.append([distance, edge_cosine_dist, (node1_index - (node2_iter + self.residue_index))])
    self.residue_index = len(self.labels[0])

  def main(self):
    protein_count = 0
    zero_count = 0
    for i in range(0, len(self.protein_tasks_files)):
      tasks_frame = pd.read_csv(self.path_data+'tasks/'+self.protein_tasks_files[i])
      features_frame = pd.read_csv(self.path_data+'features/'+self.protein_features_files[i])
      protein_frame = pd.concat([features_frame, tasks_frame.drop(['Unnamed: 0', 'annotation_sequence'], axis=1)], axis=1)
      """ Data Sanity Check: Counting NaNs and removing them """
      NaN_Count = len(tasks_frame) - tasks_frame.count()
      # self.protein_frame = protein_frame.dropna(subset=[n for n in protein_tasks_frame if n != 'annotation_domain'])
      # self.protein_frame = protein_frame.dropna(subset=[n for n in protein_frame if n != 'annotation_domain'])
      self.protein_frame = protein_frame.dropna()
      """ Exclude the empty protein files: """
      if self.protein_frame.shape[0] != 0:
        protein_count += 1
        self.protein_Nan_Count.append(NaN_Count.sum())
        # print("%s: %s" % (protein_count, self.protein_tasks_files[i].split(".", -1)[0]))
        coordinate_frame, feature_frame, output_frame, sequence_data = self.data_prep()
        self.graph_gen(coordinate_frame=coordinate_frame,
                       feature_frame=feature_frame,
                       output_frame=output_frame, sequence_data=sequence_data)
        self.protein_files_name.append(self.protein_features_files[i].split("/", -1)[-1].split('.')[0])
        self.node_num.append(self.protein_frame.shape[0])
        """ Extracting the Protein ID from the path and creating list of nodes with their protein ids:"""
        self.nodes_protein_id.append([self.protein_features_files[i].split("/", -1)[-1].split('.')[0]] * feature_frame.shape[0])
        self.nodes_protein_id_flag.append((protein_count-1) * np.ones(feature_frame.shape[0]).astype(np.int32))
        self.protein_frame = []
      else:
        zero_count += 1
        self.protein_frame = []

    self.edges = [self.edge_node1, self.edge_node2]