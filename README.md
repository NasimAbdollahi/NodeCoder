<!-- # NodeCoder Pipeline -->

<img src="/figures/Nodecoder_banner.png" width = "1070">

![APM](https://img.shields.io/apm/l/NodeCoder?style=plastic) ![GitHub package.json version](https://img.shields.io/github/package-json/v/NasimAbdollahi/NodeCoder)
![GitHub file size in bytes](https://img.shields.io/github/size/:NasimAbdollahi/:NodeCoder?style=plastic)


A PyTorch implementation of **NodeCoder Pipeline**, a Graph Convolutional Network (GCN) model for protein residue characterization. 
This work was presented at **NeurIPS MLSB 2021**: *Residue characterization on AlphaFold2 protein structures using graph neural networks*. [link to paper](https://www.mlsb.io/papers_2021/MLSB2021_Residue_characterization_on_AlphaFold2.pdf)

## Abstract:
```
Three-dimensional structure prediction tools offer a rapid means to approximate the topology of a protein structure for any
protein sequence. Recent progress in deep learning-based structure prediction has led to highly accurate predictions that have
recently been used to systematically predict 20 whole proteomes by DeepMind’s AlphaFold and the EMBL-EBI. While highly convenient,
structure prediction tools lack much of the functional context presented by experimental studies, such as binding sites or 
post-translational modifications. Here, we introduce a machine learning framework to rapidly model any residue-based
classification using AlphaFold2 structure-augmented protein representations. Specifically, graphs describing the 3D structure of
each protein in the AlphaFold2 human proteome are generated and used as input representations to a Graph Convolutional Network
(GCN), which annotates specific regions of interest based on the structural attributes of the amino acid residues, including their
local neighbors. We demonstrate the approach using six varied amino acid classification tasks.
```
<img src="/figures/NodeCoder_Pipeline.png" width = "1070">


## Table of Contents
🧬 [ What does NodeCoder Pipeline do? ](#u1)<br>
⚙️ [ Installing NodeCoder ](#u2)<br>
🔌 [ NodeCoder Usage ](#u3)<br>
🗄️ [ Graph data files ](#u4)<br> 
📂 [ Output files ](#u5)<br>
🤝 [ Collaborators ](#u6)<br>
🔐 [ License ](#u7)<br>
📄 [ Citing this work ](#u8)

<a name="u1"></a>
### 🧬 What does the NodeCoder Pipeline do? 

---
The NodeCoder is a generalized framework that annotates 3D protein structures with predicted tasks such as 
binding sites. The NodeCoder model is based on Graph Convolutional Network. NodeCoder generates proteins' graphs from 
ALphaFold2 augmented proteins' structures where the nodes are the amino acid residues and edges are inter-residue 
contacts within a preset distance. The NodeCoder model is then trained with generated graph data for users task of 
interest like: `Task = ['y_Ligand']`. 
When running inference, NodeCoder takes the **Protein ID** like **EGFR_HUMAN** and for the proteins that are already in 
the database, input graph data files are created from the AlphaFold2 protein structure in addition to calculating some 
structure-based and sequence-based residue features. The input graph data will then be given to the trained model for 
prediction of multiple tasks of interest such as binding sites or post-translational modifications.

<img src="/figures/NodeCoder_FunctionBlocks.png" width = "950">

<a name="u2"></a>
### ⚙️ Installing NodeCoder 

---
#### Required dependencies
The codebase is implemented in Python 3.8 and package versions used for development are:
```
numpy              1.19.2
pandas             1.2.4
scipy              1.6.3
torch              0.4.1
torchvision        0.9.1
torchaudio         0.8.1
torch_scatter      2.0.6
torch_sparse       0.6.9
torch_cluster      1.5.9
torch_spline_conv  1.2.0
torch-geometric    1.7.0  
scikit-learn       0.24.2
matplotlib         3.3.3
biopython          1.77
freesasa           2.0.5.post2
loguru             0.6.0

```
#### Installation steps
Here is the step-by-step NodeCoder installation process:
1. Before installing NodeCoder, we highly recommend to create a virutal Python 3.8 environment using venv command, 
or Anaconda. Assuming you have anaconda3 installed on your computer, on your 
Terminal run the following command line:
```
$ conda create -n NodeCoder_env python=3.8
```
2. Clone the repository:
```
$ git clone https://github.com/NasimAbdollahi/NodeCoder.git
```
3. Make sure your virtual environment is active. For conda environment you can use this command line: 
```
$ conda activate NodeCoder_env
```

3. Make sure you are in the root directory of the NodeCoder package `~/NodeCoder/` (where setup.py is). 
Now install NodeCoder package with following command line, which will install all dependencies in the python environment
you created in step 1:
```
$ pip install .
```

<a name="u3"></a>
### 🔌 NodeCoder Usage

---
NodeCoder package can be employed for train and inference. Here we describe how to use it:

#### 🗂️ Preprocessing raw data
[link to paper](https://www.mlsb.io/papers_2021/MLSB2021_Residue_characterization_on_AlphaFold2.pdf)
NodeCoder uses AlphaFold2 modeled protein structures as input. [AlphaFold protein structure database](https://alphafold.ebi.ac.uk/)
provides open access to protein structure predictions of human proteome and other key proteins of interest. 
Prediction labels can be obtained from [BioLip database](https://zhanggroup.org//BioLiP/qsearch_pdb.cgi?pdbid=1h88) and 
[Uniprot database](https://www.uniprot.org/).
To extract node features and labels from these databases, NodeCoder has a **featurizer** module. When using NodeCoder, 
first step after installation is to run the featurizer module. This module will create two files for every protein in 
the selected proteome: <font color='#D55E00'> *.features.csv </font> and <font color='#D55E00'> *.tasks.csv </font> . 
These files are saved in <font color='#D55E00'> NodeCoder/data/input_data/featurized_data/TAX_ID/ </font> 
directory in separate folders of <font color='#D55E00'> features </font> and <font color='#D55E00'> tasks </font>. 
The command line to run the featurizer module is:
```
$ python NodeCoder/main_preprocess_raw_data.py
```  

#### 🗃️ Generate graph data
The next step after running the featurizer is to generate graph data from the features and tasks files. NodeCoder has a 
graph-generator module that generate protein graph data by taking a threshold for distance between 
amino acid residues. The threshold distance is required to be defined by user in Angstrom unit to create the graph contact
network, `threshold_dist = 5`. Graph data files are saved in this directory <font color='#D55E00'> ./data/input_data/graph_data_*A/ </font>.
The command line to run the graph generator module is:
```
$ python NodeCoder/main_generate_graph_data.py
```  

#### 🧠 Train NodeCoder
To train NodeCoder's graph-based model, user needs to run `main_train.py` script.
Script `parser.py` has the model parameters used for training the model.
User would need to use the following parameters in `main_train.py` script to specify the task/tasks of interest and the
cutoff distance for defining the protein contact network:
``` 
Task = ['y_Ligand']
threshold_dist = 5
```
Command line to train NodeCoder:
```
$ python NodeCoder/main_train.py
```

Here is a list of available training tasks (residue labels/annotations) :
```
'y_CHAIN', 'y_TRANSMEM', 'y_MOD_RES', 'y_ACT_SITE', 'y_NP_BIND', 
'y_LIPID', 'y_CARBOHYD', 'y_DISULFID', 'y_VARIANT', 'y_Artifact', 
'y_Peptide', 'y_Nucleic', 'y_Inorganic', 'y_Cofactor', 'y_Ligand'
```

#### 🤖 Inference with NodeCoder
To use trained NodeCoder for protein functions prediction, user needs to run `main_predict.py` script.
User would need to use the following parameters in `main_predict.py` script to specify the protein of interest, functions
of interest and the cutoff distance for defining the protein contact network:
``` 
Task = ['y_Ligand']
threshold_dist = 5
```
Command line to run inference with NodeCoder:
```
$ python NodeCoder/main_predict.py
```
The user shall make sure the model with the desired parameters should have been trained already, otherwise the user would
need to first train the model then use this pipeline for prediction.


<a name="u4"></a>
### 🗄️ Graph data files

---
When graph data is generated from featurized data, files are saved in this directory <font color='#D55E00'> ./data/input_data/graph_data_*A/ </font>. 
Specific sub-directories are created depends on user choice of cutoff distance for protein contact network, proteom, 
number of cross-validation folds. This helps user to keep track of different test cases.

<details open>
<summary> Generated graph data files includes: </summary>
<br>

#### *_features.csv  <br />
Nodes' feature vectors are concatenated to create a long list of features for all nodes in the graph.
The first row is a header:

| node_id | feature_id | value |
| --- | --- | --- |
| node number in the graph | one int. value of 0-45  | ... |

#### *_edges.csv <br />
It contains the list node-pairs for all edges in the graph:

| id1 | id2 |
| --- | --- |
| index of node 1 | index of node 2 |

#### *_edge_features.csv <br />
It contains the edge features for all edges in the graph.
Currently, three different features are calculated for the edges; however, the current model only uses the squared reciprocal 
of edge length as weight. 

| id1 | id2 | edge_length | edge_cosine_angle | edge_sequence_distance |
| --- | --- | --- | --- | --- | 
| index of node 1 | index of node 2 | Euclidean distance between the node pair | Cosine distance between the node pair | Sequence distance between the node pair |

#### *_target.csv <br />
It the target labels for all nodes in the graph. The number of columns depends on the number of targets that user specify for prediction. 

| y_task1 | y_task2 | y_task3 | y_task4 | y_task5 | y_task6 | 
| --- | --- | --- | --- | --- | --- | 
| 0/1 |  0/1 | 0/1 | 0/1 | 0/1 | 0/1 |

#### *_ProteinFiles.csv <br />

| Protein File | Node Num | Removed NaNs|
| --- | --- | --- |
| EGFR_HUMAN | 1207 | 0 |
| PTC1_HUMAN | 1444 | 0 |
| DDX10_HUMAN | 872 | 0 |
| ... | ... | ... |
| RBL1_HUMAN | 1065 | 0 |


#### *_nodes_ProteinID.csv <br />
If more than one protein is given to the model, this file keeps track of the residues that belong to each protein. 

| node_id | protein_id_flag | protein_id|
| --- | --- | --- |
| 0  | 0 | EGFR_HUMAN |
| 1  | 0 | EGFR_HUMAN |
| 2  | 0 | EGFR_HUMAN |
| 3  | 0 | EGFR_HUMAN |
| 4  | 0 | EGFR_HUMAN |
| ...  | ... | ...|
| 1207  | 1 | E2F8_HUMAN |
| 1208  | 1 | E2F8_HUMAN |
| 1209  | 1 | E2F8_HUMAN |
| ...  | ... | ...|

Amino Acid Residue (AA) feature vector:

| feature ID | feature name | feature Description |
| --- | --- | --- |
| 0-19 | feat_A, feat_B, ..., feat_Y  | primary structure - one-hot-encoding binary vector of the 20 natural amino acid names |
| 20-23 | feat_PHI, feat_PSI, feat_TAU, feat_THETA | Dihedral angles, φ, ψ, τ, θ |
| 24 | feat_BBSASA | Back Bone Solvent Accessibility |
| 25 | feat_SCSASA | Side Chain Solvent Accessibility |
| 26 | feat_pLDDT | Show file differences that haven't been staged |
| 27-41 | feat_DSSP_* | Secondary structure features, e.g. α-helix and β-sheet  |
| 42 | feat_CentricDist | Euclidean distance of AA residue from the center of protein |
| 43 | feat_CentricCosineDist | Cosine distance of AA residue from the center of protein |
| 44 | feat_iPlus | AA info of a node before node i in protein sequence |
| 45 | feat_iMinus | AA info of a node after node i in protein sequence |

</details>

<a name="u5"></a>
### 📂 Output files

---
All output files are saved in this directory <font color='#D55E00'> ./results/graph_*A/ </font>. Specific sub-directories are created
according to model parameters, so that user can keep track of different test cases.

#### When training NodeCoder
In a cross-validation setting, the performance scores are saved in a .csv file like 
<font color='#D55E00'> Model_Performance_Metrics_Fold1.csv </font>, for all folds. In addition to this, model state is also saved 
in <font color='#D55E00'> CheckPoints </font> sub-directory at certain epochs. The checkpoint epoch can be specified in `parser.py`.
At the end of training on each fold, the inference is performed by finding the optimum epoch and loading corresponding 
trained model at the optimum epoch. At the end of inference, an output file is saved in <font color='#D55E00'> Prediction </font> 
sub-directory that includes the predicted labels for all proteins in validation set. This can be useful for ranking proteins.

#### When predicting with NodeCoder
When running inference with trained NodeCoder, the prediction results are saved in a sub-directory with protein name. 
The prediction result is a csv file like <font color='#D55E00'> KI3L1_HUMAN_prediction_3Tasks_results.csv </font>, which is a dataframe 
that contains the target labels, predicted labels and prediction probability of the labels per node (AA residue) for all 
tasks of interest, {y1, y2, ..., yn}.


| node_id | protein_id_flag | protein_id | Task 1 Target| Task 1 Prediction | Task 1 PredictionProb | ... | Task n Target | Task n Prediction | Task n PredictionProb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | E2F8_HUMAN | 0/1 | 0/1 | float |  | 0/1 | 0/1 | float |
| 1 | 0 | E2F8_HUMAN | 0/1 | 0/1 | float |  | 0/1 | 0/1 | float |


<a name="u6"></a>
### 🤝 Collaborators

---
This project has been sponsored by [Mitacs](https://www.mitacs.ca/en) and [Cyclica Inc.](https://www.cyclicarx.com/).  <br />
The main contributors are:  <br />
**Nasim Abdollahi**, Ph.D., Post-doctoral Fellow at University of Toronto, Cyclica Inc. <br />
**Ali Madani**, Ph.D., Machine Learning Director at Cyclica Inc. <br />
**Bo Wang**, Ph.D., Canada CIFAR AI Chair at the Vector Institute, Professor at University of Toronto <br />
**Stephen MacKinnon**, Ph.D., Chief Platform Officer at Cyclica Inc. <br />

<a name="u7"></a>
### 🔐 License
MIT Licence 

<a name="u8"></a>
### 📄 Citing this work
```
@article {2021,
	author = {Abdollahi, Nasim and Madani, Ali and Wang, Bo and MacKinnon,
	Stephen},
	title = {Residue characterization on AlphaFold2 protein structures using graph neural networks},
	year = {2021},
	doi = {},
	publisher = {NeurIPS},
	URL = {https://www.mlsb.io/papers_2021/MLSB2021_Residue_characterization_on_AlphaFold2.pdf},
	journal = {NeurIPS, MLSB}
}
```
[comment]: <> (colors = ['#CC79A7', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#000000'])
