import os
import torch
import numpy as np
import random
import multiprocessing
import time
from NodeCoder.utilities.parser import parameter_parser
from NodeCoder.gcn.NodeCoder import NodeCoder_Model
from NodeCoder.gcn.train_wrapper import Wrapper
from NodeCoder.utilities.utils import tab_printer, plot_performance_metrics
from NodeCoder.utilities.config import logger


def main(threshold_dist:int=5, multi_task_learning:bool=False, Task:list=['y_Ligand'], centrality_feature:bool=True,
           cross_validation_fold_number:int=5):
  """
  In order to run this script successfully, Separate protein graphs for train and validation should have already been
  generated.
  What does main_train_MP.py do:
  - Parsing command line parameters
  - Reading graph data (input_data/graph_data_*A/*.csv)
  - Graph decomposition
  - fitting a NodeCoder and scoring the model in multiprocessing setting.
  Model parameters can be defined in parser.py
  """

  """ 
  default setting for proteome:
  TAX_ID = '9606'
  PROTEOME_ID = 'UP000005640'
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
                          Task=Task, centrality_feature=centrality_feature, cross_validation_fold_number=cross_validation_fold_number)
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

  """ multiprocessing setting to train on multiple folds: Cross Validation """
  start_time = time.time()
  train = Wrapper(args, NodeCoder_Network)
  processes = []
  for i in range(0, args.cross_validation_fold_number):
    train_process = multiprocessing.Process(target=train.train_fold, args=[i])
    train_process.start()
    processes.append(train_process)
  for process in processes:
    process.join()

  plot_performance_metrics(args)
  logger.success(f"Training NodeCoder on all folds completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
  main()