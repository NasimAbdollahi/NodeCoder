import os
import torch
import numpy as np
import random
import time
from NodeCoder.featurizer.build_datasets import DataBuilder
from NodeCoder.utilities.parser import parameter_parser
from NodeCoder.utilities.utils import tab_printer
from NodeCoder.utilities.config import logger

def main(alphafold_data_path:str='not provided', uniprot_data_path:str='not provided', biolip_data_path:str='not provided',
         biolip_data_skip_path:str='not provided', TAX_ID:str='9606', PROTEOME_ID:str='UP000005640'):
  """
  Parsing command line parameters,
  Reading raw_data (AlphaFold, BioLip, uniprot)
  Generating features and labels from raw data and writing featurized_data: *.features.csv & *.tasks.csv in input_data
  Generating graph data and writing data files in input_data/graph_data_*A
  """

  """ default is single-task learning unless it is specified! """
  args = parameter_parser(NodeCoder_usage='data_generation', alphafold_data_path=alphafold_data_path,
                          uniprot_data_path=uniprot_data_path, biolip_data_path=biolip_data_path,
                          biolip_data_skip_path=biolip_data_skip_path, TAX_ID=TAX_ID, PROTEOME_ID=PROTEOME_ID)
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
  logger.success(f"Generating features and tasks files from raw data completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
  main()
