import tqdm
import os
from loguru import logger
from multiprocessing import Pool, cpu_count
import NodeCoder.featurizer.protein_features as protein_features
import NodeCoder.featurizer.protein_tasks as protein_tasks
import NodeCoder.featurizer.protein_sequences as protein_sequences


class DataBuilder(object):
    """
    Data_Builder builds features and labels from raw data from: AlphaFold, BioLip, uniprot
    For each protein in the data set, generated features and tasks are then saved as:
    features/*.features.csv
    tasks/*.tasks.csv
    """

    def __init__(self, args):
        self.args = args

    def generate_protein_task_df(self, protein):
        """
        Manages bulk creation of task dataframes
        """
        save_file = os.path.join(self.args.path_featurized_data_tasks, f"{protein.entry_name}.tasks.csv")
        if os.path.exists(save_file):
            logger.info(f"Skipping.  Detected pre-existing save file: {save_file}")
        else:
            try:
                known_sites = self.all_biolip_df.loc[self.all_biolip_df['Uniprot'].isin(set(protein.accessions))]
                task_df = protein_tasks.task_dataframe(protein, known_sites)
                task_df.to_csv(save_file)
                logger.info(f"Completed task df file for: {save_file}")
            except Exception:
                logger.info(f'Unable to Execute Protein {save_file}')

    def generate_protein_feature_df(self, protein):
        """
        Manages build creation of feature dataframes

        Notes on AF2 database:
        There can be multiple segments of the homology model, Numbered by the F value which appear to exist only for large
        model files > 1400.  By manually inspecting Q86UQ4, it seems that alphafold will model upto 1400 residues and when
        a protein exceeds that value, will create rolling models for windows of 200 residues, starting at 0, 200, 400, etc
        with F names of F1, F2, F3... etc.  For simplicity in this analysis, lets restrict models to F1 files.
        """
        save_file = os.path.join(self.args.path_featurized_data_features, f"{protein.entry_name}.features.csv")
        if os.path.exists(save_file):
            logger.info(f"Skipping.  Detected pre-existing save file: {save_file}")
        else:
            known_models = self.all_homology_models.loc[self.all_homology_models['accession'].isin(set(protein.accessions))]
            model_1 = known_models[known_models['file_number'] == 'F1']
            assert len(model_1) <= 1, f"More than one F1 model found for protein {known_models}"
            if len(model_1) == 1:
                try:
                    full_path = os.path.join(self.args.path_raw_data_AlphaFold, model_1.iloc[0]['file_name'])
                    feat_df = protein_features.feature_dataframe(protein, full_path)
                    feat_df.to_csv(save_file)
                    logger.info(f"Completed feature df file for: {save_file}")
                except Exception:
                    logger.info(f'Unable to Execute Protein {save_file}')

    def main(self):
        # Generate uniprot list required for task or feature dataframes and define what you want to do
        if self.args.path_raw_data_uniprot == 'not provided':
            logger.warning('the path to UniProt data is required. uniprot_data_path takes the path in str type.')
            exit()
        else:
            protein_list = protein_sequences.read_proteome(self.args.path_raw_data_uniprot)
        run_tasks = True
        run_features = True

        # Generate Task Dataframes
        if run_tasks:
            if self.args.path_raw_data_BioLiP == 'not provided' & self.args.path_raw_data_BioLiP_skip == 'not provided':
                logger.warning('the path to BioLiP data and skip are required. biolip_data_path and biolip_data_skip_path'
                               ' take the path in str type.')
                exit()
            else:
                self.all_biolip_df = protein_tasks.build_biolip_df(self.args.path_raw_data_BioLiP, self.args.path_raw_data_BioLiP_skip)
                logger.info(f'Initiating MP evaluation of {len(protein_list)} protein task dataframes with {cpu_count()} cpus.')
                with Pool(processes=cpu_count()) as pool:
                    _ = [x for x in tqdm.tqdm(pool.imap(self.generate_protein_task_df, protein_list))]

        # Generate Feature Dataframes
        if run_features:
            if self.args.path_raw_data_AlphaFold == 'not provided':
                logger.warning('the path to AlphaFold data is required. alphafold_data_path takes the path in str type.')
                exit()
            else:
                self.all_homology_models = protein_features.load_alphafold_index(self.args.path_raw_data_AlphaFold)
                logger.info(f'Structure Models Parsed\n{self.all_homology_models}')
                logger.info(f'Initiating MP evaluation of {len(protein_list)} protein feature dataframes with {cpu_count()} cpus.')
                with Pool(processes=cpu_count()) as pool:
                    _ = [x for x in tqdm.tqdm(pool.imap(self.generate_protein_feature_df, protein_list))]