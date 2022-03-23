import argparse

def parameter_parser(NodeCoder_usage:str='predict', TAX_ID:str='9606', PROTEOME_ID:str='UP000005640', Task:str='NA',
                     protein_ID:str='NA', protein_fold_number:int=0, threshold_dist:int=5, multi_task_learning:str='No'):
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")

    parser.set_defaults(NodeCoder_usage=NodeCoder_usage)

    parser.set_defaults(TAX_ID='9606')
    parser.set_defaults(PROTEOME_ID='UP000005640')
    parser.add_argument("--path-raw-data",
                        nargs="?",
                        default="./data/raw_data/",
                        help="Raw Data Path.")
    parser.add_argument("--path-raw-data-AlphaFold",
                        nargs="?",
                        default="./data/raw_data/alphafold/20210722/%s_%s/"%(PROTEOME_ID, TAX_ID),
                        help="Raw Data Path (AlphaFold).")
    parser.add_argument("--path-raw-data-uniprot",
                        nargs="?",
                        default="./data/raw_data/uniprot_proteomes/20200917/uniprot.%s.%s.2020-09-15.txt.gz"%(TAX_ID, PROTEOME_ID),
                        help="Raw Data Path (UniProt).")
    parser.add_argument("--path-raw-data-BioLiP",
                        nargs="?",
                        default="./data/raw_data/BioLiP/all_annotations.tsv",
                        help="Raw Data Path (BioLiP).")
    parser.add_argument("--path-raw-data-BioLiP-skip",
                        nargs="?",
                        default="./data/raw_data/BioLiP/ligand_list.txt",
                        help="Raw Data Path (BioLiP - SKIP).")
    parser.add_argument("--path-featurized-data",
                        nargs="?",
                        default="./data/input_data/featurized_data/%s/"%TAX_ID,
                        help="Featurized Data Path.")                    
    parser.add_argument("--path-featurized-data-tasks",
                        nargs="?",
                        default="./data/input_data/featurized_data/%s/tasks/"%TAX_ID,
                        help="Featurized Data Path (tasks).")
    parser.add_argument("--path-featurized-data-features",
                        nargs="?",
                        default="./data/input_data/featurized_data/%s/features/"%TAX_ID,
                        help="Featurized Data Path (Features).")

    parser.set_defaults(threshold_dist=threshold_dist)
    parser.add_argument("--cross-validation-fold-number",
                        type=int,
                        default=3,
                        help="Number of folds for cross-validation.")
    args = parser.parse_args()
    parser.set_defaults(KnownProteins_filename='KnownProteinFiles.csv')
    parser.add_argument("--path-graph-data",
                        nargs="?",
                        default="./data/input_data/graph_data_%sA/%sFoldCV/" %(threshold_dist, args.cross_validation_fold_number),
                        help="GraphData Path.")
    parser.add_argument("--path-results",
                        nargs="?",
                        default="./results/graph_%sA/%sFoldCV/" %(threshold_dist, args.cross_validation_fold_number),
                        help="Results Path.")
    parser.add_argument("--path-protein-results",
                        nargs="?",
                        default="./results/graph_%sA/%sFoldCV/%s/" %(threshold_dist, args.cross_validation_fold_number,protein_ID),
                        help="Protein Results Path.")

    parser.set_defaults(multi_task_learning=multi_task_learning)
    parser.set_defaults(protein_ID=protein_ID)
    parser.set_defaults(protein_fold_number=protein_fold_number)
    parser.set_defaults(graph_data_targets_name=['y_CHAIN', 'y_TRANSMEM', 'y_MOD_RES',
                                                 'y_ACT_SITE', 'y_NP_BIND', 'y_LIPID', 'y_CARBOHYD', 'y_DISULFID',
                                                 'y_VARIANT', 'y_Artifact', 'y_Peptide', 'y_Nucleic', 'y_Inorganic',
                                                 'y_Cofactor', 'y_Ligand'])
    parser.set_defaults(target_name=Task)
    parser.set_defaults(includeEdgeFeature='Yes')
    parser.set_defaults(downSampling_majority_class='No')
    parser.set_defaults(downSampling_majority_class_ratio=0.6)
    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="Physical", #"Physical", "metis", "random"
	                    help="Clustering method for graph decomposition. Default is the Physical procedure.")
    parser.add_argument("--train-cluster-number",
                        type=int,
                        default=1,
                        help="Number of train clusters extracted. Default is 10.")
    parser.add_argument("--validation-cluster-number",
                        type=int,
                        default=1,
                        help="Number of validation clusters extracted. Default is 10.")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
	                    help="Number of training epochs. Default is 200.")
    parser.add_argument("--PerformanceStep",
                        type=int,
                        default=4,
                        help="Epochs where performance metrics are calculated. Must be greater than one. Default is 10.")
    parser.add_argument("--CheckPointStep",
                        type=int,
                        default=4,
                        help="Epochs where calculated model weights and metrics are saved. Default is 10.")
    parser.add_argument("--seed",
                        type=int,
                        default=10,
	                    help="Random seed for train-test split. Default is 10.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                    help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                    help="Learning rate. Default is 0.01.")
    parser.add_argument("--betas",
                        type=float,
                        default=(0.9, 0.999),
                        help="betas for adam optimizer. Default is (0.9, 0.999).")
    parser.add_argument("--weighted-loss",
                        type=str,
                        default='non', # 'non', 'Logarithmic', 'Power_Logarithmic','Linear','Smoothed_Linear' , 'Sigmoid'
                        help="weighted loss scheme.")
    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.01,
	                    help="Test data ratio. Default is 0.2.")
    parser.set_defaults(input_layers=[38, 28, 18, 8])
    parser.set_defaults(output_layer_size=8)

    args = parser.parse_args()

    train_edge_path, train_edge_feature_path, train_features_path, train_target_path, train_protein_filename_path = [], [], [], [], []
    for i in range(0, args.cross_validation_fold_number):
        filename_edge = 'train_'+str(i+1)+'_edges.csv'
        filename_edge_feature = 'train_'+str(i+1)+'_edge_features.csv'
        filename_feature = 'train_'+str(i+1)+'_features.csv'
        filename_target = 'train_'+str(i+1)+'_target.csv'
        filename_ProteinFile = 'train_'+str(i+1)+'_ProteinFiles.csv'
        train_edge_path.append('%s%s' %(args.path_graph_data, filename_edge))
        train_edge_feature_path.append('%s%s' %(args.path_graph_data, filename_edge_feature))
        train_features_path.append('%s%s' %(args.path_graph_data, filename_feature))
        train_target_path.append('%s%s' %(args.path_graph_data, filename_target))
        train_protein_filename_path.append('%s%s' %(args.path_graph_data, filename_ProteinFile))
    parser.set_defaults(train_edge_path=train_edge_path)
    parser.set_defaults(train_edge_feature_path=train_edge_feature_path)
    parser.set_defaults(train_features_path=train_features_path)
    parser.set_defaults(train_target_path=train_target_path)
    parser.set_defaults(train_protein_filename_path=train_protein_filename_path)

    validation_edge_path, validation_edge_feature_path, validation_features_path, validation_target_path, validation_protein_filename_path, validation_node_proteinID_path = [], [], [], [], [], []
    for i in range(0, args.cross_validation_fold_number):
        filename_edge = 'validation_'+str(i+1)+'_edges.csv'
        filename_edge_feature = 'validation_'+str(i+1)+'_edge_features.csv'
        filename_feature = 'validation_'+str(i+1)+'_features.csv'
        filename_target = 'validation_'+str(i+1)+'_target.csv'
        filename_ProteinFile = 'validation_'+str(i+1)+'_ProteinFiles.csv'
        filename_Node_ProteinID = 'validation_'+str(i+1)+'_nodes_ProteinID.csv'
        validation_edge_path.append('%s%s' %(args.path_graph_data, filename_edge))
        validation_edge_feature_path.append('%s%s' %(args.path_graph_data, filename_edge_feature))
        validation_features_path.append('%s%s' %(args.path_graph_data, filename_feature))
        validation_target_path.append('%s%s' %(args.path_graph_data, filename_target))
        validation_protein_filename_path.append('%s%s' %(args.path_graph_data, filename_ProteinFile))
        validation_node_proteinID_path.append('%s%s' %(args.path_graph_data, filename_Node_ProteinID))
    parser.set_defaults(validation_edge_path=validation_edge_path)
    parser.set_defaults(validation_edge_feature_path=validation_edge_feature_path)
    parser.set_defaults(validation_features_path=validation_features_path)
    parser.set_defaults(validation_target_path=validation_target_path)
    parser.set_defaults(validation_protein_filename_path=validation_protein_filename_path)
    parser.set_defaults(validation_node_proteinID_path=validation_node_proteinID_path)

    if protein_ID != 'NA':
        filename_edge = protein_ID + '_edges.csv'
        filename_edge_feature = protein_ID + '_edge_features.csv'
        filename_feature = protein_ID + '_features.csv'
        filename_target = protein_ID + '_target.csv'
        filename_ProteinFile = protein_ID + '_ProteinFiles.csv'
        filename_Node_ProteinID = protein_ID + '_nodes_ProteinID.csv'
        prediction_fileName = protein_ID + '_prediction_'+ str(len(args.target_name))+'Tasks_results.csv'
        parser.set_defaults(protein_edge_path='%s%s' %(args.path_protein_results, filename_edge))
        parser.set_defaults(protein_edge_feature_path='%s%s' %(args.path_protein_results, filename_edge_feature))
        parser.set_defaults(protein_features_path='%s%s' %(args.path_protein_results, filename_feature))
        parser.set_defaults(protein_target_path='%s%s' %(args.path_protein_results, filename_target))
        parser.set_defaults(protein_filename_path='%s%s' %(args.path_protein_results, filename_ProteinFile))
        parser.set_defaults(protein_node_proteinID_path='%s%s' %(args.path_protein_results, filename_Node_ProteinID))
        parser.set_defaults(protein_prediction_fileName='%s%s' %(args.path_protein_results, prediction_fileName))

    if args.NodeCoder_usage == 'train':
        if args.multi_task_learning == 'No':
            filename = args.target_name[0]+'_HiddenLayers_'+str(args.input_layers)+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)
        else:
            filename = str(len(args.target_name))+'Targets_HiddenLayers_'+str(args.input_layers)+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)

        CheckPoint_path, Metrics_path, Metrics_tasks_path, Metrics_clusters_tasks_path, Prediction_fileName, Prediction_Metrics_fileName = [], [], [], [], [], []
        for i in range(0, args.cross_validation_fold_number):
            filename_CheckPoint = filename + '/CheckPoints/Fold'+str(i+1)+'/'
            filename_Metrics = filename + '/Model_Performance_Metrics_Fold'+str(i+1)+'.csv'
            filename_Metrics_task = filename + '/Model_Performance_Metrics_tasks_Fold'+str(i+1)+'.csv'
            filename_Metrics_task_cluster = filename + '/Model_Performance_Metrics_clusters_tasks_Fold'+str(i+1)+'.csv'
            filename_Prediction = filename + '/Prediction/Final_Prediction_Fold'+str(i+1)+'.csv'
            filename_Prediction_Metrics = filename + '/Prediction/Model_Performance_Final_Prediction_Metrics_Fold'+str(i+1)+'.csv'
            CheckPoint_path.append('%s%s' %(args.path_results, filename_CheckPoint))
            Metrics_path.append('%s%s' %(args.path_results, filename_Metrics))
            Metrics_tasks_path.append('%s%s' %(args.path_results, filename_Metrics_task))
            Metrics_clusters_tasks_path.append('%s%s' %(args.path_results, filename_Metrics_task_cluster))
            Prediction_fileName.append('%s%s' %(args.path_results, filename_Prediction))
            Prediction_Metrics_fileName.append('%s%s' %(args.path_results, filename_Prediction_Metrics))
        parser.set_defaults(CheckPoint_path=CheckPoint_path)
        parser.set_defaults(Metrics_path=Metrics_path)
        parser.set_defaults(Metrics_tasks_path=Metrics_tasks_path)
        parser.set_defaults(Metrics_clusters_tasks_path=Metrics_clusters_tasks_path)
        parser.set_defaults(Prediction_fileName=Prediction_fileName)
        parser.set_defaults(Prediction_Metrics_filename=Prediction_Metrics_fileName)
        parser.set_defaults(Prediction_path=args.path_results + filename + '/Prediction/')

    else:
        CheckPoint_path, Metrics_path, Prediction_fileName, Prediction_Metrics_fileName = [], [], [], [],
        for t in args.target_name:
            filename = t+'_HiddenLayers_'+str(args.input_layers)+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)
            filename_CheckPoint = filename + '/CheckPoints/Fold'+str(protein_fold_number)+'/'
            filename_Metrics = filename + '/Model_Performance_Metrics_Fold'+str(protein_fold_number)+'.csv'
            CheckPoint_path.append('%s%s' %(args.path_results, filename_CheckPoint))
            Metrics_path.append('%s%s' %(args.path_results, filename_Metrics))
        parser.set_defaults(CheckPoint_path=CheckPoint_path)
        parser.set_defaults(Metrics_path=Metrics_path)

    return parser.parse_args()