import argparse


def parameter_parser(NodeCoder_usage:str, alphafold_data_path:str='not provided', uniprot_data_path:str='not provided',
                     biolip_data_path:str='not provided', biolip_data_skip_path:str='not provided', TAX_ID:str='9606',
                     PROTEOME_ID:str='UP000005640', Task:str='NA', protein_ID:str='NA', trained_model_fold_number:int=1,
                     threshold_dist:int=5, cross_validation_fold_number:int=5, multi_task_learning:bool=False,
                     centrality_feature:bool=False, epochs:int=1000, checkpoint_step:int=50, performance_step:int=50,
                     learning_rate:float=0.01, network_layers:list=[38, 28, 18, 8], weighted_loss:str='non',
                     train_ratio:float=0.8, train_cluster_number:int=1, validation_cluster_number:int=1):
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")

    parser.set_defaults(NodeCoder_usage=NodeCoder_usage)
    parser.set_defaults(centrality_feature=centrality_feature)

    parser.set_defaults(TAX_ID=TAX_ID)
    parser.set_defaults(PROTEOME_ID=PROTEOME_ID)
    parser.set_defaults(path_raw_data_AlphaFold=alphafold_data_path)
    parser.set_defaults(path_raw_data_uniprot=uniprot_data_path)
    parser.set_defaults(path_raw_data_BioLiP=biolip_data_path)
    parser.set_defaults(path_raw_data_BioLiP_skip=biolip_data_skip_path)

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
    parser.add_argument("--path-test-featurized-data",
                        nargs="?",
                        default="./data/test_data/featurized_data/%s/"%TAX_ID,
                        help="Test Data Path.")

    parser.set_defaults(threshold_dist=threshold_dist)
    parser.set_defaults(cross_validation_fold_number=cross_validation_fold_number)
    args = parser.parse_args()
    parser.set_defaults(KnownProteins_filename='KnownProteinFiles.csv')
    parser.add_argument("--path-graph-data",
                        nargs="?",
                        default="./data/input_data/graph_data_%sA/%s/%sFoldCV/" %(threshold_dist, TAX_ID, args.cross_validation_fold_number),
                        help="GraphData Path.")
    parser.add_argument("--path-results",
                        nargs="?",
                        default="./results/graph_%sA/%s/%sFoldCV/" %(threshold_dist, TAX_ID, args.cross_validation_fold_number),
                        help="Results Path.")
    parser.add_argument("--path-protein-results",
                        nargs="?",
                        default="./results/graph_%sA/%s/%sFoldCV/%s/" %(threshold_dist, TAX_ID, args.cross_validation_fold_number,protein_ID),
                        help="Protein Results Path.")

    parser.set_defaults(multi_task_learning=multi_task_learning)
    parser.set_defaults(protein_ID=protein_ID)
    parser.set_defaults(trained_model_fold_number=trained_model_fold_number)
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
    parser.set_defaults(train_cluster_number=train_cluster_number)
    parser.set_defaults(validation_cluster_number=validation_cluster_number)
    parser.set_defaults(epochs=epochs)
    parser.set_defaults(performance_step=performance_step)
    parser.set_defaults(checkpoint_step=checkpoint_step)
    parser.add_argument("--seed",
                        type=int,
                        default=10,
                        help="Random seed for train-test split. Default is 10.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout parameter. Default is 0.5.")
    parser.set_defaults(learning_rate=learning_rate)
    parser.add_argument("--betas",
                        type=float,
                        default=(0.9, 0.999),
                        help="betas for adam optimizer. Default is (0.9, 0.999).")
    parser.set_defaults(weighted_loss=weighted_loss)
    # parser.add_argument("--weighted-loss",
    #                     type=str,
    #                     default='non', # 'non', 'Logarithmic', 'Power_Logarithmic','Linear','Smoothed_Linear' , 'Sigmoid'
    #                     help="weighted loss scheme.")
    parser.set_defaults(train_ratio=train_ratio)
    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.01,
                        help="Test data ratio. Default is 0.2.")
    parser.set_defaults(input_layers=network_layers)
    args = parser.parse_args()
    parser.set_defaults(output_layer_size=args.input_layers[-1])


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
        hidden_layers = '_'.join([str(l) for l in args.input_layers])
        if args.multi_task_learning:
            filename = str(len(args.target_name))+'Targets_HiddenLayers_'+hidden_layers+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)
        else:
            filename = args.target_name[0]+'_HiddenLayers_'+hidden_layers+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)

        CheckPoint_path, Metrics_path, Metrics_tasks_path, Metrics_clusters_tasks_path, Prediction_fileName, Prediction_Metrics_fileName = [], [], [], [], [], []
        for i in range(0, args.cross_validation_fold_number):
            filename_CheckPoint = filename + '/CheckPoints/Fold'+str(i+1)+'/'
            filename_Metrics = filename + '/Model_Performance_Metrics_Fold'+str(i+1)+'.csv'
            filename_Metrics_task = filename + '/Model_Performance_Metrics_tasks_Fold'+str(i+1)+'.csv'
            filename_Metrics_task_cluster = filename + '/Model_Performance_Metrics_clusters_tasks_Fold'+str(i+1)+'.csv'
            filename_Prediction = filename + '/Prediction/Final_Prediction_Fold'+str(i+1)+'.csv'
            filename_Prediction_Metrics = filename + '/Prediction/Model_Performance_Metrics_PerProtein_Fold'+str(i+1)+'.csv'
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
        hidden_layers = '_'.join([str(l) for l in args.input_layers])
        if args.multi_task_learning:
            filename = str(len(args.target_name))+'Targets_HiddenLayers_'+hidden_layers+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)
            filename_CheckPoint = filename + '/CheckPoints/Fold'+str(trained_model_fold_number)+'/'
            filename_Metrics = filename + '/Model_Performance_Metrics_Fold'+str(trained_model_fold_number)+'.csv'
            CheckPoint_path.append('%s%s' %(args.path_results, filename_CheckPoint))
            Metrics_path.append('%s%s' %(args.path_results, filename_Metrics))
        else:
            for t in args.target_name:
                filename = t+'_HiddenLayers_'+hidden_layers+'_'+str(args.epochs)+'Epochs_LR'+str(args.learning_rate)
                filename_CheckPoint = filename + '/CheckPoints/Fold'+str(trained_model_fold_number)+'/'
                filename_Metrics = filename + '/Model_Performance_Metrics_Fold'+str(trained_model_fold_number)+'.csv'
                CheckPoint_path.append('%s%s' %(args.path_results, filename_CheckPoint))
                Metrics_path.append('%s%s' %(args.path_results, filename_Metrics))
        parser.set_defaults(CheckPoint_path=CheckPoint_path)
        parser.set_defaults(Metrics_path=Metrics_path)

    return parser.parse_args()