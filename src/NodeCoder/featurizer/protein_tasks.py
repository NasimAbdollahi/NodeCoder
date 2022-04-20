import collections
import pandas as pd
import NodeCoder.featurizer.protein_sequences as protein_sequences
from Bio import SwissProt
from loguru import logger


def build_biolip_df(biolip_file: str, biolip_skip: str) -> pd.DataFrame:
    """
    Reads a flat file from BioLip that indexes binding sites found on pdb files and cross-references to biolip_sequence
    :param biolip_file: flat file of pockets downloaded from https://zhanglab.ccmb.med.umich.edu/BioLiP/
    :param biolip_skip: biolip file of artifact het codes
    :return: A dataframe of annotated binding sites that are mappable onto other sequences
    note: There are more predictive options that are skipped here (catalytic residues, ec numbers, go terms)
    """

    # Prepare Ligand Types
    logger.info(f"Loading Biolip File: {biolip_file}")
    inorganic = {'ZN', 'CA', 'MG', 'MN', 'FE', 'CU', 'SF4', 'FE2', 'CO', 'FES', 'CO3', 'CU', '', '', }
    cofactors = {'CLA', 'HEM', 'HEC', 'BCL'}
    skip_ligands = set()
    with open(biolip_skip) as F:
        for het in F.readlines():
            skip_ligands.add(het.strip())

    # Read Biolip File
    counter_types = collections.defaultdict(int)
    counter_ligs = collections.defaultdict(int)
    parsed_ligands = []
    with open(biolip_file) as F:
        for line in F:
            fields = line.strip('\n').split('\t')
            pdb_id = f"{fields[0]}_{fields[1]}"
            lig_het = fields[4].strip()
            uniprot = fields[17]
            sequence = fields[19]
            residues = biolip_binding_site_parser(fields[8], sequence)
            if lig_het in skip_ligands:
                biolip_type = 'Artifact'
            elif lig_het == 'III':
                biolip_type = 'Peptide'
            elif lig_het == 'NUC':
                biolip_type = 'Nucleic'
            elif lig_het in inorganic:
                biolip_type = 'Inorganic'
            elif lig_het in cofactors:
                biolip_type = 'Cofactor'
            elif len(residues) == 0:
                biolip_type = 'Error'
            else:
                biolip_type = 'Ligand'
                counter_ligs[lig_het] += 1
            counter_types[biolip_type] += 1
            if residues is not None:
                parsed_ligands.append((pdb_id, lig_het, uniprot, biolip_type, sequence, residues, fields[8]))

    # Report Most Common Types and Ligands
    top_ligs = sorted(counter_ligs.items(), key=lambda y: y[1], reverse=True)
    top_types = sorted(counter_types.items(), key=lambda y: y[1], reverse=True)
    logger.info(f"Top Types: {top_types}")
    logger.info(f"Top Ligands: {top_ligs[0:100]}")
    headers = ['PDB_Chain', 'Het', 'Uniprot', 'Type', 'Sequence', 'Residues', 'Raw']
    biolip_df = pd.DataFrame(parsed_ligands, columns=headers)
    logger.info(biolip_df)
    return biolip_df


def biolip_binding_site_parser(binding_residues: str, biolip_sequence: str) -> list:
    """
    Parses biolip binding sites to a representation that is mappable onto a seqres record
    :param binding_residues: string field 7 from the biolip pocket flat file denoting binding residue name and numbers
                         eg: 'H57 Y60A W60D L99 I174 D189 A190 C191 E192 S195 S214 W215 G216 E217 G219 R221A G226'
    :param biolip_sequence: the re-indexed biolip_sequence provided by biolip
    :return: A list of the sequence indices corresponding to the binding site of interest, empty if there is an error
    """

    residue_ids = []
    for residue in binding_residues.split():
        aa = residue[0]
        raw_number = residue[1:]
        try:
            residue_number = int(raw_number) - 1
        except ValueError:
            residue_number = int(raw_number[0:-1]) - 1
        if not biolip_sequence[residue_number] == aa:
            logger.error(f"Unmapped Residue {residue_number}->{aa} does not match sequence {biolip_sequence}")
            return []
        residue_ids.append(residue_number)
    return residue_ids


def task_dataframe(uniprot_object: SwissProt.Record, binding_sites: pd.DataFrame) -> pd.DataFrame:
    """
    Maps annotations to protein sequences to build the task dataframe
    :param uniprot_object: swissprot protein object to obtain sequence and annotations
    :param binding_sites: Subselection of protein binding sites extracted from biolip_df
    :return task dataframe where each row is a residue and each column is a task
    """

    # Build a dictionary dataframe to get started, one-hot encode sequence
    labels = {'annotation_sequence': list(uniprot_object.sequence)}
    n = len(uniprot_object.sequence)

    # Now add the feature labels from the uniprot, can add more feature types
    feature_types = ['CHAIN', 'TRANSMEM', 'MOD_RES', 'ACT_SITE', 'NP_BIND', 'LIPID', 'CARBOHYD', 'DISULFID', 'VARIANT']
    feature_type_set = set(feature_types)
    for feature_type in feature_types:
        labels[f'y_{feature_type}'] = [False] * n
    for feature in uniprot_object.features:
        if feature.type in feature_type_set:
            try:
                if feature.type == 'DISULFID':
                    labels[f"y_{feature.type}"][feature.location.start] = True
                    labels[f"y_{feature.type}"][feature.location.end-1] = True
                else:
                    for i in range(feature.location.start, feature.location.end):
                        labels[f"y_{feature.type}"][i] = True
            except TypeError:
                logger.warning(f'Detected a feature with unknown position: {uniprot_object.entry_name} - {feature}')
            except IndexError:
                logger.warning(f'Detected a feature with an invalid position: {uniprot_object.entry_name} - {feature}')

    # Binding Sites
    for binding_type in ['Artifact', 'Peptide', 'Nucleic', 'Inorganic', 'Cofactor', 'Ligand']:
        labels[f'y_{binding_type}'] = [False] * len(uniprot_object.sequence)
    for site in binding_sites.iloc:
        uni_to_biolip, biolip_to_uniprot = protein_sequences.sequence_sequence_mapping(uniprot_object.sequence,
                                                                                       site['Sequence'])
        for aa_index in site['Residues']:
            ref_index = biolip_to_uniprot[aa_index]
            labels[f'y_{site["Type"]}'][ref_index] = True

    # Create dataframe and return
    return pd.DataFrame(labels)
