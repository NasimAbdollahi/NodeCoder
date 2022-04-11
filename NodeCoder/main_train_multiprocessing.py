import os
import torch
import numpy as np
import random
import multiprocessing
import time
from utilities.parser import parameter_parser
from gcn.NodeCoder import NodeCoder_Model
from gcn.train_wrapper import Wrapper
from utilities.utils import tab_printer, plot_performance_metrics
from utilities.config import logger


def main():
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
  Here you need to specify:
  Tasks of interest
  Threshold distance in Angstrom (A) for creating graph contact network 
  """
  Task = ['y_Ligand']
  threshold_dist = 5

  """ default is single-task learning unless it is specified! """
  args = parameter_parser(NodeCoder_usage='train', threshold_dist=threshold_dist, multi_task_learning=False, Task=Task)
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