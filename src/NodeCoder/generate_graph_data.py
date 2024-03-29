import os
import torch
import numpy as np
import random
import multiprocessing
import time
from NodeCoder.utilities.parser import parameter_parser
from NodeCoder.utilities.utils import tab_printer
from NodeCoder.graph_generator.graph_data_generator import Graph_Data_Generator
from NodeCoder.utilities.config import logger


def main(TAX_ID:str='9606', PROTEOME_ID:str='UP000005640', threshold_dist:int=5, cross_validation_fold_number:int=1):
  """
  In order to run this script successfully, featurized data is required to be generated by running the
  main_preprocessing_raw_data.py.
  What does main_graph_data.py do:
  - Parsing command line parameters
  - Reading featurized_data (input_data/featurized_data/*.features.csv & input_data/featurized_data/*.tasks.csv)
  - Generating graph data and writing data files in: input_data/graph_data_*A
  - Generating graph data of multiple folds are done in multiprocessing setting.
  """

  """
  TAX_ID = '9606'
  PROTEOME_ID = 'UP000005640'
  """

  """ 
  you need to specify Threshold distance in Angstrom (A) for creating graph contact network
  threshold_dist = 5
  """

  """ default is single-task learning unless it is specified! """
  args = parameter_parser(NodeCoder_usage='graph_data_generation', TAX_ID=TAX_ID, PROTEOME_ID=PROTEOME_ID,
                          threshold_dist=threshold_dist, cross_validation_fold_number=cross_validation_fold_number)
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
  graph_data = Graph_Data_Generator(args)
  graph_data.grouping_proteins_for_train_validation_folds()
  processes = []
  for i in range(0, args.cross_validation_fold_number):
    train_graph = multiprocessing.Process(target=graph_data.train_graph_data_files_generator, args=[i])
    train_graph.start()
    processes.append(train_graph)
    validation_graph = multiprocessing.Process(target=graph_data.validation_graph_data_files_generator, args=[i])
    validation_graph.start()
    processes.append(validation_graph)
  for process in processes:
    process.join()

  logger.success(f"Generating all graph data files completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
  main()
