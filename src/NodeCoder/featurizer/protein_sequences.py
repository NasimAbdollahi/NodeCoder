import Bio.PDB
import Bio.PDB.Chain
import Bio.SeqUtils
from Bio import SwissProt
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as MatList
import gzip
from loguru import logger
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

matrix = MatList.blosum62
gap_open = -10
gap_extend = -0.5


def sequence_sequence_mapping(sequence1: str, sequence2: str) -> Tuple[list, list]:
    """
    Creates a pair of index maps between two sequences
    :param sequence1: first sequence to align
    :param sequence2: second sequence to align
    :return: A pair of lists for mapping sequence 1 indices to sequence 2 indexes and vice versa
    """
    seq1_map = [-1] * len(sequence1)
    seq2_map = [-1] * len(sequence2)
    aln_m, aln_d, score, begin, end = pairwise2.align.globalds(sequence1, sequence2, matrix, gap_open, gap_extend)[0]
    seq1_ind = 0
    seq2_ind = 0
    for s, t in zip(aln_m, aln_d):
        if not t == '-' and not s == '-':
            seq1_map[seq1_ind] = seq2_ind
            seq2_map[seq2_ind] = seq1_ind
        if not s == '-':
            seq1_ind += 1
        if not t == '-':
            seq2_ind += 1
    return seq1_map, seq2_map


def sequence_structure_mapping(sequence: str, chain_object: Bio.PDB.Chain.Chain) -> Union[Tuple[Dict, Dict], None]:
    """
    Creates a map of indices from a biopython object chain onto a uniprot and vice versa
    :param sequence: Canonical sequence to map residue numbers onto
    :param chain_object: Biopython chain representing the structure to map
    :return:  A pair of lists for mapping sequence indices to structure indexes and vice versa
    """
    sequence_to_structure = {}
    structure_to_sequence = {}
    ppb = Bio.PDB.PPBuilder()
    peptides = ppb.build_peptides(chain_object)
    for p in peptides:
        pseq = p.get_sequence()
        if not str(pseq) in sequence:
            logger.error(f'Modeled Segment does not match uniprot {pseq} in {sequence}')
            return None
        prnm = map(lambda r: r.get_parent().id[1], p.get_ca_list())
        pfst = [-1] * len(pseq)
        try:
            aln_m, aln_d, score, begin, end = pairwise2.align.globalds(sequence, pseq, matrix, gap_open, gap_extend)[0]
        except SystemError:
            logger.error(f"Alignment Error: {pseq} onto {sequence}")
            return None
        fasta_ind = 0
        atomr_ind = 0
        for s, t in zip(aln_m, aln_d):
            if not t == '-' and not s == '-':
                pfst[atomr_ind] = fasta_ind
            if not s == '-':
                fasta_ind += 1
            if not t == '-':
                atomr_ind += 1
        for resname, atomr, refseq in zip(pseq, prnm, pfst):
            sequence_to_structure[refseq] = atomr
            structure_to_sequence[atomr] = refseq
    return sequence_to_structure, structure_to_sequence


def read_proteome(proteome_file: str, test_max=None) -> List:
    """
    Creates a map of indices from a biopython object chain onto a uniprot and vice versa
    :param proteome_file: Uniprot SP format file containing all proteins of interest (ie proteome)
    :param test_max: Integer maximum number of proteins to read for purposes of code testing
    """
    logger.info(f"Reading UniProt Files: {proteome_file}")
    with gzip.open(proteome_file, 'rt') as F:
        protein_list = []
        for sp_object in SwissProt.parse(F):
            protein_list.append(sp_object)
            if test_max and len(protein_list) >= test_max:
                break
    logger.info(f"Identified {len(protein_list)} total proteins for processing.")
    return protein_list
