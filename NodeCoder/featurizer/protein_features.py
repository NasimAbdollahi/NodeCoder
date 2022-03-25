import Bio.PDB
import Bio.PDB.Chain
import Bio.SeqUtils
import subprocess
import os
import pandas as pd
import freesasa
import gzip
from Bio import SwissProt
from Bio.PDB import DSSP
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Model import Model
from loguru import logger
from tempfile import NamedTemporaryFile
import NodeCoder.featurizer.protein_sequences as protein_sequences
freesasa.setVerbosity(freesasa.nowarnings)


def load_alphafold_index(alphafold_folder: str) -> pd.DataFrame:
    """
    Summarizes content from an folder with an unpacked alphafold proteome tar file
    :param alphafold_folder: Directory of alphafold pdb files
    :return: formatted dataframe of models which can be used for representation
    """
    logger.info(f"Reading Alphafold Files: {alphafold_folder}")
    model_files = [x for x in os.listdir(alphafold_folder) if x.endswith('.pdb.gz')]
    logger.info(f"  -> Parsed {len(model_files)}; Sample 3: {model_files[0:3]}")
    logger.info(f"  -> Total Unique Uniprots {len(set(model_files))}")

    model_file_df = pd.DataFrame([x.strip('.pdb.gz').split('-') for x in model_files],
                                 columns=['tag', 'accession', 'file_number', 'model_code'])
    model_file_df['file_name'] = model_files
    logger.info(f"  -> Generated Model Dataframe:\n{model_file_df}")
    return model_file_df


def sasa_occupancy(pdb_entity: Bio.PDB.Chain.Chain):
    """
    Populates the 'occupancy' column with SASA in Å² via the occupancy field in the biopython atom object
    :param pdb_entity: Any biopython entity with atoms
    """

    # Now we run freesasa and load the results in the occupancy field of the atom
    asa_engine = freesasa.Structure()
    atom_list = [atom for atom in pdb_entity.get_atoms()]
    for i in range(0, len(atom_list)):
        atom = atom_list[i]
        name = f'{atom.element:>2s}'
        resid = (str(atom.parent.id[1])+atom.parent.id[2]).strip()
        (x, y, z) = atom.coord
        asa_engine.addAtom(name, atom.parent.resname, resid, atom.parent.parent.id, x, y, z)
    if asa_engine.nAtoms() > 0:
        result = freesasa.calc(asa_engine)
        for i in range(0, len(atom_list)):
            atom = atom_list[i]
            atom.set_occupancy(result.atomArea(i))


def sequence_features(uniprot_object: SwissProt.Record, pdb_chain: Bio.PDB.Chain.Chain, dssp: DSSP) -> pd.DataFrame:
    """
    Examines the feature records of a protein to generate structure-based features
    :param uniprot_object: Swissprot record parsed by SwissProt.parse(fileIO)
    :param pdb_chain: bio.pdb chain object with 3d coordinates corresponding to the relevant chain
    :param dssp: DSSP object from the dssp_from_filename object below
    :return: sequence dataframe with columns corresponding to observed features
    """

    # Build a dictionary dataframe to get started, one-hot encode sequence
    labels = {'annotation_sequence': list(uniprot_object.sequence)}
    n = len(uniprot_object.sequence)
    for aa_type in 'ACDEFGHIKLMNPQRSTVWY':
        labels[f'feat_{aa_type}'] = [False] * n
    for i, aa in enumerate(uniprot_object.sequence):
        try:
            labels[f"feat_{aa}"][i] = True
        except KeyError:
            logger.warning(f"Non-standard AA {aa} in sequence {uniprot_object.sequence}")

    # Now for sasa, and atomreq biolip_sequence and phi, psi, omega
    ppb = PPBuilder()
    sasa_occupancy(pdb_chain)
    mapping = protein_sequences.sequence_structure_mapping(uniprot_object.sequence, pdb_chain)
    if mapping is None:
        logger.warning(f"Unable to map atomrec onto uniprot for {uniprot_object.entry_name}. Skipping.")
        return None
    seq_to_atom, atom_to_seq = mapping
    backbone = ['CA', 'C', 'O', 'N']
    labels['annotation_atomrec'] = [None] * n
    for feature in ['feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT']:
        labels[feature] = [0] * n
    for d_char in ['H', 'B', 'E', 'G', 'I', 'T', 'S']:
        labels[f'feat_DSSP_{d_char}'] = [False] * n
    for d_prop in range(6, 14):
        labels[f'feat_DSSP_{d_prop}'] = [0] * n

    for coord in ['coord_X', 'coord_Y', 'coord_Z']:
        labels[coord] = [0] * n
    no_ss = 0
    all_ss = 0
    for peptide in ppb.build_peptides(pdb_chain):
        seqs = peptide.get_sequence()
        taus = peptide.get_tau_list()
        thetas = peptide.get_theta_list()
        phis, psis = zip(*peptide.get_phi_psi_list())
        for res, aa, phi, psi, tau, theta in zip(peptide, seqs, phis, psis, taus, thetas):
            ref_index = atom_to_seq[res.id[1]]
            sasa = sum([x.occupancy for x in res])
            bb_sasa = sum([res[x].occupancy for x in backbone])
            sc_sasa = sasa - bb_sasa
            labels['annotation_atomrec'][ref_index] = aa
            labels['feat_PHI'][ref_index] = phi if phi else 0
            labels['feat_PSI'][ref_index] = psi if psi else 0
            labels['feat_TAU'][ref_index] = tau if tau else 0
            labels['feat_THETA'][ref_index] = theta if theta else 0
            labels['feat_BBSASA'][ref_index] = bb_sasa
            labels['feat_SCSASA'][ref_index] = sc_sasa
            labels['coord_X'][ref_index] = res['CA'].coord[0]
            labels['coord_Y'][ref_index] = res['CA'].coord[1]
            labels['coord_Z'][ref_index] = res['CA'].coord[2]
            labels['feat_pLDDT'][ref_index] = res['CA'].bfactor
            try:
                dssp_res = dssp[(res.parent.id, res.id)]
                for d_char in ['H', 'B', 'E', 'G', 'I', 'T', 'S']:
                    labels[f'feat_DSSP_{d_char}'][ref_index] = d_char == dssp_res[2]
                for d_prop in range(6, 14):
                    labels[f'feat_DSSP_{d_prop}'][ref_index] = dssp_res[d_prop]
            except KeyError:
                no_ss += 1
            all_ss += 1

    # Create dataframe and return
    return pd.DataFrame(labels)


def dssp_from_filename(model: Model, full_path: str) -> DSSP:
    """
    A wrapper to get biopython's dssp working with alphafold files.  DSSP has some unnecessary pdb files formatting
    restrictions that require me to remove the model and endmdl lines, and the biopython DSSP parser isn't behaving well
    so i mimic'd direct submission of a dssp file.
    :param full_path: absolute filepath of alphafold pdb file
    :param model: corresponding biopython model object for the relevant ss assignment
    """

    with NamedTemporaryFile(suffix='.pdb') as temp_pdb:
        with gzip.open(full_path, 'rt') as PDB_FILE:
            for line in PDB_FILE:
                if line.startswith('ENDMDL'):
                    temp_pdb.write(b'END')
                    break
                if not line.startswith('MODEL'):
                    temp_pdb.write(line.encode())
        p = subprocess.Popen(["mkdssp", temp_pdb.name], universal_newlines=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    with NamedTemporaryFile(suffix='.dssp') as temp_dssp:
        temp_dssp.write(out.encode())
        dssp_obj = DSSP(model, temp_dssp.name, file_type='DSSP')
    return dssp_obj


def feature_dataframe(uniprot_object: SwissProt.Record, model_path: str) -> pd.DataFrame:
    """
    Generates a feature dataframe for the given uniprot object and saves it to disk.
    :param uniprot_object: swissprot protein object
    :param model_path: full disk path of the model file to featurize
    :return feature dataframe where each row is a residue and each column is a feature or coordinates to build network
    """

    with gzip.open(model_path, 'rt') as F:
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('????', F)
    dssp = dssp_from_filename(structure[0], model_path)
    chains = sorted(structure[0].child_list, key=lambda y: len(y), reverse=True)
    assert len(chains) == 1  # The SM version sometimes had bound peptides and we defaulted to longest seg
    chain = chains[0]        # For alphafold however, there should only be one chain and no empty dataframes
    return sequence_features(uniprot_object, chain, dssp)
