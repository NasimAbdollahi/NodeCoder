import numpy as np
import pandas as pd
import os
import math
import random
import glob
import time
from NodeCoder.graph_generator.protein_graph import protein_graph_generator
from NodeCoder.utilities.utils import csv_writter_known_proteins, csv_writter_grouping_protein, csv_writter_graph_data
from NodeCoder.utilities.config import logger

class Graph_Data_Generator(object):
    """
    Creating two separate graphs for train and validation.
    csv_path: the path to save generated graphs.
    protein_count: the size of the graphs can be determined by number of proteins.
    threshold: distance cut-off based on euclidean distance between c-alpha atoms
    """
    def __init__(self, args):
        """
        """
        self.args = args
        if not os.path.exists(self.args.path_graph_data):
            os.makedirs(self.args.path_graph_data, exist_ok=True)

    def find_known_proteins(self):
        """
        Creates a list of all protein files with known 3D structures.
        """
        if not os.path.exists(self.args.path_featurized_data+self.args.KnownProteins_filename):
            protein_files = glob.glob(self.args.path_featurized_data+'/tasks/*.tasks.csv')
            known_protein_files = []
            node_num = []
            missing_features_files = []
            for i in range(0, len(protein_files)):
                protein_frame = pd.read_csv(protein_files[i])
                protein_frame = protein_frame.dropna()
                if protein_frame.shape[0] != 0:
                    output_frame = protein_frame.filter(regex='y_').astype(int)
                    y_Peptide_count = np.count_nonzero(output_frame['y_Peptide'])
                    y_Ligand_count = np.count_nonzero(output_frame['y_Ligand'])
                    y_Nucleic_count = np.count_nonzero(output_frame['y_Nucleic'])
                    #y_Artifact_count = np.count_nonzero(output_frame['y_Artifact'])
                    y_Cofactor_count = np.count_nonzero(output_frame['y_Cofactor'])
                    y_Inorganic_count = np.count_nonzero(output_frame['y_Inorganic'])
                    Known_Features = [y_Peptide_count, y_Ligand_count, y_Nucleic_count, y_Cofactor_count, y_Inorganic_count]
                    if any(Known_Features):
                        features_file_path = protein_files[i].replace('tasks', 'features')
                        if os.path.exists(features_file_path):
                            known_protein_files.append(protein_files[i].split("/", -1)[-1])
                            node_num.append(protein_frame.shape[0])
                        else:
                            missing_features_files.append(protein_files[i].replace('tasks', 'features'))
            csv_writter_known_proteins(known_protein_files, node_num, self.args.path_featurized_data, self.args.KnownProteins_filename)
            known_proteins = pd.read_csv(self.args.path_featurized_data+self.args.KnownProteins_filename)
        else:
            known_proteins = pd.read_csv(self.args.path_featurized_data+self.args.KnownProteins_filename)
        return known_proteins

    def grouping_proteins_for_train_validation_folds(self):
        """
        This module reads a list of known proteins.
        Then creates groups of train and validation proteins from for cross validation setting.
        Total of 3823 proteins in data Dec2021 are Known Human Proteins.
        In 5fold_CV setting there are 765 proteins in each validation fold.
        """
        protein_filenames = list(self.find_known_proteins()["tasks file"])
        if self.args.cross_validation_fold_number == 1:
            protein_number_validation = math.floor(len(protein_filenames)*(1-self.args.train_ratio))
            protein_number_train = len(protein_filenames) - protein_number_validation
        else:
            protein_number_validation = math.floor(len(protein_filenames)/self.args.cross_validation_fold_number)
            protein_number_train = len(protein_filenames) - protein_number_validation

        logger.info("Total number of proteins with known 3D structures (from AlphaFold Dec2021Data):")
        logger.info(f"Total number of proteins selected for training:{protein_number_train}")
        logger.info(f"Total number of proteins selected for validation: {protein_number_validation}")

        validation_protein_filenames = []
        train_protein_filenames = []
        for i in range(0, self.args.cross_validation_fold_number):
            Remained_protein_filenames = list.copy(protein_filenames[0:len(protein_filenames)])
            if self.args.cross_validation_fold_number == 1:
                """ for no cross-validation case, default is %80 for train and %20 for validation: """
                random.shuffle(Remained_protein_filenames)
                validation_protein_filenames.append(Remained_protein_filenames[0:protein_number_validation])
                del Remained_protein_filenames[0:protein_number_validation]
                train_protein_filenames.append(Remained_protein_filenames)
            else:
                """ for cross-validation case: """
                if i == self.args.cross_validation_fold_number-1:
                    validation_protein_filenames.append(Remained_protein_filenames[i*protein_number_validation:])
                    del Remained_protein_filenames[i*protein_number_validation:]
                else:
                    validation_protein_filenames.append(Remained_protein_filenames[i*protein_number_validation:(i+1)*protein_number_validation])
                    del Remained_protein_filenames[i*protein_number_validation:(i+1)*protein_number_validation]
                train_protein_filenames.append(Remained_protein_filenames)
            """ write csv files: """
            validation_name = 'validation_' + str(i+1)
            train_name = 'train_' + str(i+1)
            csv_writter_grouping_protein(self.args.path_graph_data, validation_name, validation_protein_filenames[i])
            csv_writter_grouping_protein(self.args.path_graph_data, train_name, train_protein_filenames[i])

    def train_graph_data_files_generator(self, i):
        """
        generates and writes train graph data files for all folds.
        to avoid regenerating the graph data, we first check if the graph data is generated or not.
        """
        # self.grouping_proteins_for_train_validation_folds()
        if not os.path.exists(self.args.path_graph_data+'train_'+str(i+1)+'_nodes_ProteinID.csv'):
            start_time = time.time()
            logger.info(f"Generating train graph data for fold {i+1} started.")

            Train_FilePath = self.args.path_graph_data + 'train_' + str(i+1) + '_ProteinFileNames.csv'
            train_protein_tasks_filenames = np.array(pd.read_csv(Train_FilePath)["tasks file"])
            train_protein_features_filenames = np.array(pd.read_csv(Train_FilePath)["features file"])
            train_protein_graph = protein_graph_generator(self.args.path_featurized_data, protein_tasks_files=train_protein_tasks_filenames, protein_features_files=train_protein_features_filenames,
                                                          target_output=self.args.graph_data_targets_name, threshold_distance=self.args.threshold_dist)
            train_protein_graph.main()
            train_name = 'train_' + str(i+1)
            csv_writter_graph_data(train_protein_graph, train_name, self.args.graph_data_targets_name, self.args.path_graph_data)
            logger.success(f"Generating train graph data for fold {i+1} completed in {time.time() - start_time} seconds.")
        else:
            logger.success(f"Train graph data files for fold {i+1} have already been generated.")

    def validation_graph_data_files_generator(self, i):
        """
        generates and writes validation graph data files for all folds.
        to avoid regenerating the graph data, we first check if the graph data is generated or not.
        """
        if not os.path.exists(self.args.path_graph_data+'validation_'+str(i+1)+'_nodes_ProteinID.csv'):
            start_time = time.time()
            logger.info(f"Generating validation graph data for fold {i+1} started ...")
            Validation_FilePath = self.args.path_graph_data + 'validation_' + str(i+1) + '_ProteinFileNames.csv'
            validation_protein_tasks_filenames = np.array(pd.read_csv(Validation_FilePath)["tasks file"])
            validation_protein_features_filenames = np.array(pd.read_csv(Validation_FilePath)["features file"])
            validation_protein_graph = protein_graph_generator(self.args.path_featurized_data, protein_tasks_files=validation_protein_tasks_filenames, protein_features_files=validation_protein_features_filenames,
                                                               target_output=self.args.graph_data_targets_name, threshold_distance=self.args.threshold_dist)
            validation_protein_graph.main()
            validation_name = 'validation_' + str(i+1)
            csv_writter_graph_data(validation_protein_graph, validation_name, self.args.graph_data_targets_name, self.args.path_graph_data)
            logger.success(f"Generating validation graph data for fold {i+1} completed in {time.time() - start_time} seconds.")
        else:
            logger.success(f"Validation graph data files for fold {i+1} have already been generated.")

    def protein_graph_data_files_generator(self):
        if not os.path.exists(self.args.path_protein_results + self.args.protein_ID +'_edges.csv'):
            logger.info("Generating protein graph data started ...")
            os.makedirs(self.args.path_protein_results, exist_ok=True)
            protein_tasks_filename = [self.args.protein_ID + '.tasks.csv']
            protein_features_filename = [self.args.protein_ID + '.features.csv']
            protein_graph = protein_graph_generator(self.args.path_test_featurized_data, protein_tasks_files=protein_tasks_filename, protein_features_files=protein_features_filename,
                                                    target_output=self.args.graph_data_targets_name, threshold_distance=self.args.threshold_dist)
            protein_graph.main()
            csv_writter_graph_data(protein_graph, self.args.protein_ID, self.args.graph_data_targets_name, self.args.path_protein_results)
            logger.success("Generating protein graph completed.")
        else:
            logger.success("Protein graph data has already been generated.")