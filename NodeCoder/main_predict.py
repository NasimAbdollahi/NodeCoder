import os
import torch
import numpy as np
import random
import time
from parser import parameter_parser
from graph_data_generator import Graph_Data_Generator
from clustering import Clustering
from NodeCoder import NodeCoder_Model
from NodeCoder_predict import NodeCoder_Predictor
from utils import colors, tab_printer, graph_reader, feature_reader, edge_feature_reader, target_reader, optimum_epoch,\
    csv_writer_prediction


def main():
    """
    Parsing command line parameters, reading protein data, generating protein graph data, graph decomposition,
    performing prediction by loading trained models and scoring the prediction.
    Model parameters can be defined in parser.py
    """

    """ 
    Here you need to specify:
    Protein ID
    Fold number: For model trained on CrossValidation setup (The fold that this protein is in validation set not train!) 
    Tasks of interest
    Threshold distance for creating graph contact network
    """
    protein_ID = 'KI3L1_HUMAN'
    protein_fold_number = 3
    Task = ['y_Ligand', 'y_Inorganic', 'y_MOD_RES']
    threshold_dist = 5

    """ When using NodeCoder to predict protein functions, default is loading trained model with single-task learning setup! """
    args = parameter_parser(Task, protein_ID=protein_ID, protein_fold_number=protein_fold_number, threshold_dist=threshold_dist)
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
    Generate protein graph.
    This class save generated graph data and avoid regenerating them.
    """
    start_time = time.time()
    Graph_Data = Graph_Data_Generator(args)
    Graph_Data.protein_graph_data_files_generator()
    print("--- %s seconds for generating protein graph data files. ---" %(time.time() - start_time))

    """ NodeCoder Model """
    NodeCoder_Network = NodeCoder_Model(args)

    """ Preparing graph data """
    start_time = time.time()
    print(colors.HEADER + "\n--- clustering protein graph started ..." + colors.ENDC)
    protein_graph = graph_reader(args.protein_edge_path)
    protein_edge_features = edge_feature_reader(args.protein_edge_feature_path)
    protein_features = feature_reader(args.protein_features_path, args.protein_edge_path)
    protein_target = target_reader(args.protein_target_path, args.target_name)
    protein_graph_data = Clustering(args, args.protein_filename_path, protein_graph, protein_features, protein_edge_features, protein_target)
    protein_graph_data.decompose()
    print("--- %s seconds for clustering protein graph ---" %(time.time() - start_time))

    TrueLabel = []
    PredictedLabel = []
    PredictedProb = []
    for t in range(0, len(Task)):
        print(colors.HEADER + "\n--- Prediction with trained NodeCoder for %s started ... " %Task[t].split('_')[-1] + colors.ENDC)
        """ using the saved checkpoint to run inference with trained model """
        """ find the optimum epoch for reading the trained  model: """
        checkpoint_epoch = optimum_epoch(args.Metrics_path[t])
        print("--- Best epoch - NodeCoder trained for task %s: %s ---" %(Task[t], checkpoint_epoch))
        predictor = NodeCoder_Predictor(args, NodeCoder_Network.model, protein_graph_data, t, checkpoint_epoch)
        predictor.test()

        TrueLabel.append(list(predictor.targets))
        PredictedLabel.append(list(predictor.predictions))
        PredictedProb.append(list(predictor.predictions_prob))

    """ Writing predicted and target labels """
    print(colors.HEADER + "\n--- Writing predicted labels ... " + colors.ENDC)
    csv_writer_prediction(Task, TrueLabel, PredictedLabel, PredictedProb, args.protein_node_proteinID_path, args.protein_prediction_fileName)
    print(colors.HEADER + "\n--- Prediction is successfully completed! --- " + colors.ENDC)

if __name__ == "__main__":
    main()
