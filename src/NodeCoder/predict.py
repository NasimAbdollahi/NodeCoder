import os
import torch
import numpy as np
import random
import time
from NodeCoder.utilities.parser import parameter_parser
from NodeCoder.graph_generator.graph_data_generator import Graph_Data_Generator
from NodeCoder.graph_generator.clustering import Clustering
from NodeCoder.gcn.NodeCoder import NodeCoder_Model
from NodeCoder.gcn.NodeCoder_predict import NodeCoder_Predictor
from NodeCoder.utilities.utils import tab_printer, graph_reader, feature_reader, edge_feature_reader, target_reader, optimum_epoch, \
    csv_writer_prediction, csv_writer_performance_metrics_perprotein
from NodeCoder.utilities.config import logger


def main(protein_ID:str, threshold_dist:int=5, trained_model_fold_number:int=1, multi_task_learning:bool=False,
         Task:list=['y_Ligand'], centrality_feature:bool=True, cross_validation_fold_number:int=5, epochs:int=2000,
         performance_step:int=50, checkpoint_step:int=50, learning_rate:float=0.01, network_layers:list=[38, 28, 18, 8],
         train_ratio:float=0.8, train_cluster_number:int=1, validation_cluster_number:int=1):
    """
    Parsing command line parameters, reading protein data, generating protein graph data, graph decomposition,
    performing prediction by loading trained models and scoring the prediction.
    Model parameters can be defined in parser.py
    """

    """ 
    Here User needs to specify:
    1 - Protein ID: protein_ID = 'KI3L1_HUMAN'
    2 - Fold number: For model trained on CrossValidation setup (The fold that this protein is in validation set not train!) 
    3 - Tasks of interest:  ['y_Ligand', 'y_TRANSMEM'] 
    4 - Threshold distance for creating graph contact network: threshold_dist = 5
    """

    """ 
    When using NodeCoder to predict protein functions, default is loading trained model with single-task learning setup!
    Option 1: use NodeCoder for predicting different tasks using models that are trained separately, which is recommended. 
    In this case, you specify the tasks of interest as Task = ['y_Ligand', 'y_Peptide'] and multi_task_learning=False.
    Option 2: use a single trained model with multi-task learning setup by for your tasks of interest, e.g. 
    Task = ['y_Ligand', 'y_Peptide'] and set multi_task_learning=True. 
    """
    args = parameter_parser(NodeCoder_usage='predict', protein_ID=protein_ID, trained_model_fold_number=trained_model_fold_number,
                            threshold_dist=threshold_dist, multi_task_learning=multi_task_learning, Task=Task,
                            centrality_feature=centrality_feature, cross_validation_fold_number=cross_validation_fold_number,
                            epochs=epochs, performance_step=performance_step, checkpoint_step=checkpoint_step, learning_rate=learning_rate,
                            network_layers=network_layers, train_ratio=train_ratio, train_cluster_number=train_cluster_number,
                            validation_cluster_number=validation_cluster_number)
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
    logger.success(f"generating protein graph data files completed in {(time.time() - start_time)} seconds.")

    """ NodeCoder Model """
    NodeCoder_Network = NodeCoder_Model(args)
    logger.success("NodeCoder architecture initialization done.")

    """ Preparing graph data """
    start_time = time.time()
    logger.info(f"clustering protein graph started...")
    protein_graph = graph_reader(args.protein_edge_path)
    protein_edge_features = edge_feature_reader(args.protein_edge_feature_path)
    protein_features = feature_reader(args.protein_features_path, args.protein_edge_path, args.centrality_feature)
    protein_target = target_reader(args.protein_target_path, args.target_name)
    protein_graph_data = Clustering(args, args.protein_filename_path, protein_graph, protein_features, protein_edge_features, protein_target)
    protein_graph_data.decompose()
    logger.info(f"clustering protein graph completed in {(time.time() - start_time)} seconds.")

    TrueLabel = []
    PredictedLabel = []
    PredictedProb = []
    if args.multi_task_learning:
        logger.info(f"Inference with trained NodeCoder (with multi-task learning setting) started ...")
        """ using the saved checkpoint to run inference with trained model """
        """ find the optimum epoch for reading the trained model with multi-task learning setting: """
        checkpoint_epoch = optimum_epoch(args.Metrics_path[0])
        logger.info(f"Best epoch - trained NodeCoder with multi-task learning setting: {checkpoint_epoch}")
        for t in range(0, len(Task)):
            predictor = NodeCoder_Predictor(args, NodeCoder_Network.model, protein_graph_data, t, checkpoint_epoch)
            predictor.test()
            TrueLabel.append(list(predictor.targets))
            PredictedLabel.append(list(predictor.predictions))
            PredictedProb.append(list(predictor.predictions_prob))
    else:
        for t in range(0, len(Task)):
            logger.info(f"Inference with trained NodeCoder for {Task[t].split('_')[-1]} started ...")
            """ using the saved checkpoint to run inference with trained model """
            """ find the optimum epoch for reading the trained model: """
            checkpoint_epoch = optimum_epoch(args.Metrics_path[t])
            logger.info(f"Best epoch - trained NodeCoder for task {Task[t]}: {checkpoint_epoch}")
            predictor = NodeCoder_Predictor(args, NodeCoder_Network.model, protein_graph_data, t, checkpoint_epoch)
            predictor.test()
            TrueLabel.append(list(predictor.targets))
            PredictedLabel.append(list(predictor.predictions))
            PredictedProb.append(list(predictor.predictions_prob))

    """ Writing predicted and target labels """
    logger.info("Writing predicted labels ...")
    csv_writer_prediction(args.NodeCoder_usage, Task, TrueLabel, PredictedLabel, PredictedProb, args.protein_node_proteinID_path, args.protein_prediction_fileName)

    """ Writing prediction metrics per protein """
    logger.info(f"Writing prediction metrics per protein ...")
    csv_writer_performance_metrics_perprotein(predictor, args.trained_model_fold_number)

    logger.success("Inference has successfully completed.")

if __name__ == "__main__":
    main()