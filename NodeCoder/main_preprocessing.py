import os
import torch
import numpy as np
import random
import time
from featurizer.build_datasets import DataBuilder
from parser import parameter_parser
from graph_data_generator import Graph_Data_Generator
from utils import tab_printer

def main():
  """
  Parsing command line parameters,
  Reading raw_data (AlphaFold, BioLip, uniprot)
  Generating features and labels from raw data and writing featurized_data: *.features.csv & *.tasks.csv in input_data
  Generating graph data and writing data files in input_data/graph_data_*A
  """

  TAX_ID = '9606'
  PROTEOME_ID = 'UP000005640'

  """ 
  Here you need to specify:
  Threshold distance for creating graph contact network
  """
  threshold_dist = 5 #(A)

  """ default is single-task learning unless it is specified! """
  args = parameter_parser(NodeCoder_usage='data_generation', TAX_ID=TAX_ID, PROTEOME_ID=PROTEOME_ID, threshold_dist=threshold_dist)
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
  Building features and labels from raw data from: AlphaFold, BioLip, uniprot
  Generated features and tasks are then saved in input_data/featurized_data directory. For each protein in the data set 
  we will have: 
  *.features.csv in features folder and 
  *.tasks.csv in tasks folder
  """
  start_time = time.time()
  featurized_data = DataBuilder(args)
  featurized_data.main()
  print("--- %s seconds for generating features and tasks files from raw data. ---" %(time.time() - start_time))

  """ 
  Generate separate protein graphs for train and validation.
  This class save generated graph data and avoid regenerating them.
  """
  start_time = time.time()
  graph_data = Graph_Data_Generator(args)
  graph_data.graph_data_files_generator()
  print("--- %s seconds for generating graph data files. ---" %(time.time() - start_time))

if __name__ == "__main__":
  main()
