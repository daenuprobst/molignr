#!/usr/bin/env python

###############################################################
# Calculates the Universal Activation Index for class A GPCRs
###############################################################
# Author: David Wifling (david.wifling@ur.de)
###############################################################


import os
import argparse
from io import StringIO
from sys import platform
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Data.SCOPData import protein_letters_3to1 as aa3to1
from Bio.PDB import PDBParser, FastMMCIFParser
from Bio.PDB.Polypeptide import is_aa


def parse_structure(del_res_0, del_res_1):
    """Parse a PDB/cif structure"""

    name, ext = os.path.splitext(structure)
    if not ext and (len(name) == 4):
        try:
            urlretrieve('https://files.rcsb.org/download/' + name.lower() + '.pdb', sample_structure_file)
        except HTTPError as e:
            print('The PDB ID {} does not exist on the RCSB PDB server'.format(name.upper()))
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach the RCSB PDB server.')
            print('Reason: ', e.reason)
        else:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure(name, sample_structure_file)
    else:
        if not os.path.isfile(structure):
            raise IOError('File not found: {}'.format(structure))
        else:
            if structure.endswith(('pdb', 'ent')):
                parser = PDBParser(QUIET=True)
            elif structure.endswith('cif'):
                parser = FastMMCIFParser(QUIET=True)
            else:
                raise Exception('Format {} not supported. Must be .pdb/.ent/.cif'.format(ext))
            struct = parser.get_structure(name, structure)

    # Extract the first model and the specified chain of the .pdb/.ent/.cif file
    try:
        struct = struct[0][chain]
    except KeyError:
        raise Exception('Chain {} not found in the structure'.format(chain))

    # Exclude the specified non-GPCR (e.g. T4L, nanobody) residue number range from the calculation
    for res_id in range(int(del_res_0), int(del_res_1), 1):
        if struct.__contains__(res_id):
           struct.__delitem__(res_id)

    return struct


def get_pdb_sequence(struct):
    """Return the sequence of the given structure object"""

    _aainfo = lambda r: (r.id[1], aa3to1.get(r.resname, 'X'))
    seq = [_aainfo(r) for r in struct.get_residues() if is_aa(r)]

    return seq


def get_res_num_name_pairs(sequence, begin):
    """Return residue number/name pairs"""

    pairs = []
    for i, n in enumerate(sequence, start=begin):
        pairs.append((i, n))

    return pairs


def align_sequences():
    """Align the reference and sample sequences (MUSCLE) and return the mapping (hH4R: sample)"""

    global res_map
    if not os.path.isfile(muscle):
        raise IOError('Muscle not found. Please specify the muscle path with -muscle')
    muscle_cline = MuscleCommandline(muscle, clwstrict=True, profile=1, in1=sample_sequence_file, in2=ref_sequences_file)
    stdout, stderr = muscle_cline()
    alns = AlignIO.read(StringIO(stdout), "clustal")
    for aln in alns:
        if aln.id == "sample":
            aligned_A = list(aln)
        elif aln.id == "hrh4-human":
            aligned_B = list(aln)
    res_map = {}
    aa_i_A, aa_i_B = 0, 0
    for aln_i, (aa_aln_A, aa_aln_B) in enumerate(zip(aligned_A, aligned_B)):
        if aa_aln_A == '-':
            if aa_aln_B != '-':
                aa_i_B += 1
        elif aa_aln_B == '-':
            if aa_aln_A != '-':
                aa_i_A += 1
        else:
            assert sample_sequence[aa_i_A][1] == aa_aln_A
            assert hH4R_sequence[aa_i_B][1] == aa_aln_B
            res_map[hH4R_sequence[aa_i_B][0]] = sample_sequence[aa_i_A][0]
            aa_i_A += 1
            aa_i_B += 1


def get_ai_index():
    """Calculate both, residue/atom numbers/distances used for calculating the Universal Activation Index, and the Index itself"""

    assert len(hH4R_res[0]) == len(hH4R_res[1]) == len(hH4R_res_most_cons[0]) == len(hH4R_res_most_cons[1])
    global sample_res_most_cons, sample_res, sample_atoms_plain, sample_dist, index
    sample_res_most_cons, sample_diff, sample_res, sample_atoms, sample_atoms_plain, sample_dist = [[], []], [[], []], [[], []], [[], []], [[], []], []
    
    for i in range(len(hH4R_res[0])):
        for y in range(len(hH4R_res)):
            sample_diff[y].append(hH4R_res[y][i] - hH4R_res_most_cons[y][i])
            if sample_res_most_cons_manual[y][i]:
                sample_res_most_cons[y].append(sample_res_most_cons_manual[y][i])
                sample_res[y].append(sample_res_most_cons[y][i] + sample_diff[y][i])
            else:
                sample_res_most_cons[y].append(res_map.get(hH4R_res_most_cons[y][i]))
                if sample_res_most_cons[y][i] and sample_structure.__contains__(sample_res_most_cons[y][i] + sample_diff[y][i]):
                    sample_res[y].append(sample_res_most_cons[y][i] + sample_diff[y][i])
                else:
                    sample_res[y].append(res_map.get(hH4R_res[y][i]))
                    if sample_res[y][i]:
                        sample_res_most_cons[y][i] = sample_res[y][i] - sample_diff[y][i]
                        print("Warning: In TM{}, the most conserved amino acid ({}) did not match in the alignment. "
                              "But the residue {} matched.\nPlease check the prediction of the most conserved "
                              "amino acid {} and use the -TM{}_50 option in the case of an incorrect prediction.\n"
                              .format(res_TM[y][i], res_most_cons_ball[y][i], res_ball[y][i], res_most_cons_ball[y][i], res_TM[y][i]))
                    else:
                        print("Error: In TM{}, neither the most conserved amino acid ({}) nor the residue {} "
                              "matched in the alignment.\n Hence, please use the -chain and/or -del_res options. "
                              "It could be also necessary to use the -TM{}_50 option.\n".
                              format(res_TM[y][i], res_most_cons_ball[y][i], res_ball[y][i], res_TM[y][i]))
            sample_atoms[y].append(sample_structure[sample_res[y][i]]['CA'])
            sample_atoms_plain[y].append(sample_atoms[y][i].get_serial_number())
        sample_dist.append(sample_atoms[0][i] - sample_atoms[1][i])
    index = -14.43 * sample_dist[0] - 7.62 * sample_dist[1] + 9.11 * sample_dist[2] - 6.32 * sample_dist[3] - 5.22 * sample_dist[4] + 278.88


def helix_trans(v):
    return [[v[0][0]], [v[0][1]], [v[1][1], v[0][2]], [v[1][2]], [v[0][3]], [v[1][3], v[0][4]], [v[1][4], v[1][0]]]


def verbose(ss):
    if v_verbose:
        """Return TM-based residue mapping (hH4R: sample)"""
        print("Residue mapping (hH4R: {}):".format(structure.upper()), end='')
        for TM in range(7):
            print("\n{:3}\t:\t".format('TM' + str(TM + 1)), end='')
            for key, val in res_map.items():
                if TM_begin[TM] <= key <= TM_end[TM]:
                    print("{}{}: {}{}, ".format(hH4R_sequence_dict[key], key, sample_sequence_dict[val],val), end='')

        """Return residue and atom numbers used for calculating the Universal Activation Index"""
        print("\n\nResidue and atom numbers for {}: ".format(structure.upper()))
        col_labels = ['conserved', 'residues used for AI calculation', 'atoms']
        for count in range(2):
            for TM in range(7):
                count_var = 0
                for var in [sample_res_most_cons, sample_res, sample_atoms_plain]:
                    out_helix = helix_trans(var)
                    len_max = 3 if structure.endswith('cif') else len(str(max([y for x in out_helix for y in x])))
                    spaces = [6 + len_max, 3 + 2 * (13 + len_max), 3 + 2 * len_max]
                    if count == 0 and TM == 0:
                        if var == sample_res_most_cons:
                            print("{:^{len}}\t||\t".format('TM', len=3), end='')
                        print("{:^{len}}\t||\t".format(col_labels[count_var], len=spaces[count_var]), end='')
                    elif count == 1:
                        if var == sample_res_most_cons:
                            print("{:3}\t||\t".format('TM' + str(TM + 1)), end='')
                            for i_2, num_2 in enumerate(ss):
                                if out_helix[TM][0] == num_2[0]:
                                    i = i_2
                                    break
                            print("{:4}: {:1}{:<{len}}\t||\t".format(str(TM + 1) + '.50', ss[i][1], out_helix[TM][0],
                                                                     len=len_max), end='')
                        elif var == sample_atoms_plain:
                            if structure.endswith('cif'):
                                print("   N/A   \t||\t", end='')
                            else:
                                for num, num_1 in enumerate(out_helix[TM]):
                                    if len(out_helix[TM]) == 1:
                                        print("{:<{len}}\t||\t".format(num_1, len=spaces[count_var]), end='')
                                    elif len(out_helix[TM]) == 2 and num == 0:
                                        print("{:<{len}} | ".format(num_1, len=len_max), end='')
                                    elif len(out_helix[TM]) == 2 and num == 1:
                                        print("{:<{len}}\t||\t".format(num_1, len=len_max), end='')
                        else:
                            for num, (num_1, ball) in enumerate(zip(out_helix[TM], helix_trans(res_ball)[TM])):
                                for i_2, num_2 in enumerate(ss):
                                    if num_1 == num_2[0]:
                                        i = i_2
                                        break
                                if num_1 >= ss[-1][0]:
                                    print("{:4}: {:1}{:1} {:1}{:<{len}}   ".format(ball, ss[i - 2][1], ss[i - 1][1],
                                                                                   ss[i][1], num_1, len=len_max),
                                          end='')
                                elif num_1 + 1 >= ss[-1][0]:
                                    print("{:4}: {:1}{:1} {:1}{:<{len}} {:1} ".format(ball, ss[i - 2][1], ss[i - 1][1],
                                                                                      ss[i][1], num_1, ss[i + 1][1],
                                                                                      len=len_max), end='')
                                else:
                                    print(
                                        "{:4}: {:1}{:1} {:1}{:<{len}} {:1}{:1}".format(ball, ss[i - 2][1], ss[i - 1][1],
                                                                                       ss[i][1], num_1, ss[i + 1][1],
                                                                                       ss[i + 2][1], len=len_max),
                                        end='')
                                if len(out_helix[TM]) == 1:
                                    print("{:<{len}}\t||\t".format('', len=16 + len_max), end='')
                                elif len(out_helix[TM]) == 2 and num == 0:
                                    print(" | ".format(''), end='')
                                elif len(out_helix[TM]) == 2 and num == 1:
                                    print("\t||\t".format(''), end='')
                    count_var += 1
                if var == sample_atoms_plain and not (TM != 0 and count == 0):
                    print("")

        """Return distances used for calculating the Universal Activation Index"""
        print("\nDistances for {}:".format(structure.upper()), end=' ')
        for count in range(5):
            print("{}-{}: {:.{}f} Ang. | ".format(res_ball[0][count], res_ball[1][count], sample_dist[count], 2), end='')
        print("\n")

    """Return a warning if the most conserved amino acids did not reflect the typical amino acid names"""
    count = 0
    for TM in range(7):
        for i_2, num_2 in enumerate(ss):
            if helix_trans(sample_res_most_cons)[TM][0] == num_2[0]:
                i = i_2
                break
        if ss[i][1] != helix_trans(res_most_cons_name)[TM][0]:
            print("Warning: The most conserved amino acid in TM{} ({}) does not "
                  "reflect the typical amino acid name {}, but {}.".
                  format(str(TM + 1), helix_trans(res_most_cons_ball)[TM][0], helix_trans(res_most_cons_name)[TM][0], ss[i][1]))
            count += 1
    if 0 < count <= 3:
        print("Warning: The names of the most conserved amino acids did not match in {} out of 7 helices.\n"
              "Based on sequence identities between 79 and 99%, this can be normal. But we still recommended "
              "to manually validate the correctness of the predictions. In the case of incorrect predictions, "
              "please use the corresponding -TM(1-7)_50 options.\n".format(count))
    elif count > 3:
        print("Error: The names of the most conserved amino acids did not match in {} out of 7 helices.\n"
              "Hence, it is most likely that either a non-GPCR chain was selected or improper, non-GPCR "
              "(e.g. T4L, nanobody) residues were excluded. Please use the -chain and/or -del_res options.\n".format(count))

    """Return the Universal Activation Index"""
    print("Universal Activation Index for {}: {:.{}f}".format(args.structure.upper(), index, 2))


if __name__ == '__main__':
    """Command line parser"""
    muscle_default = 'muscle.exe' if platform == 'win32' else './muscle'
    ap = argparse.ArgumentParser(description='Calculates the Universal Activation Index for class A GPCRs')
    ap.add_argument("-structure", help="(Crystal) structure path (.pdb/.ent/.cif) or RCSB PDB ID (XXXX)", required=True)
    ap.add_argument("-chain", help="Chain name of the structure that contains the GPCR (default: A)", default='A')
    ap.add_argument("-del_res", help="Exclude the specified non-GPCR (e.g. T4L, nanobody) residue number range of the selected chain "
                                     "from the alignment (default: 1000-2000, disable with: 0-0)", default='1000-2000')
    ap.add_argument("-ref", help="Path for the reference sequences file (default: ref_sequences.fasta)", default='ref_sequences.fasta')
    ap.add_argument("-muscle", help="Path for the MUSCLE binary (default: muscle.exe (Windows), muscle (other OS)", default=muscle_default)
    ap.add_argument("-verbose", "-v", help="Detailed information on residue mapping/residues/atoms/distances", action="store_true", default=False)
    ap.add_argument("-TM1_50", type=int, help="Residue number of the most conserved amino acid in TM1 (1.50) (optional)", action='store')
    ap.add_argument("-TM2_50", type=int, help="Residue number of the most conserved amino acid in TM2 (2.50) (optional)", action='store')
    ap.add_argument("-TM3_50", type=int, help="Residue number of the most conserved amino acid in TM3 (3.50) (optional)", action='store')
    ap.add_argument("-TM4_50", type=int, help="Residue number of the most conserved amino acid in TM4 (4.50) (optional)", action='store')
    ap.add_argument("-TM5_50", type=int, help="Residue number of the most conserved amino acid in TM5 (5.50) (optional)", action='store')
    ap.add_argument("-TM6_50", type=int, help="Residue number of the most conserved amino acid in TM6 (6.50) (optional)", action='store')
    ap.add_argument("-TM7_50", type=int, help="Residue number of the most conserved amino acid in TM7 (7.50) (optional)", action='store')
    args = ap.parse_args()

    """Initialize variables"""
    structure = args.structure
    chain = args.chain
    del_res = args.del_res.split('-')
    ref_sequences_file = args.ref
    muscle = args.muscle
    v_verbose = args.verbose
    sample_res_most_cons_manual = [[args.TM1_50, args.TM2_50, args.TM3_50, args.TM5_50, args.TM6_50], [args.TM7_50, args.TM3_50, args.TM4_50, args.TM6_50, args.TM7_50]]
    sample_sequence_file = 'sample_sequence.fasta'
    sample_structure_file = 'sample_structure.pdb'

    """hH4R residues used for calculating the Activation Index of the sample structure"""
    hH4R_res = [[36, 61, 104, 202, 326], [360, 99, 132, 302, 340]]
    res_ball = [['1.53', '2.50', '3.42', '5.66', '6.58'], ['7.55', '3.37', '4.42', '6.34', '7.35']]

    """Most conserved amino acids in each TM used as a reference point in the alignment"""
    hH4R_res_most_cons = [[33, 61, 112, 186, 318], [355, 112, 140, 318, 355]]
    res_TM = [['1', '2', '3', '5', '6'], ['7', '3', '4', '6', '7']]
    res_most_cons_ball = [['1.50', '2.50', '3.50', '5.50', '6.50'], ['7.50', '3.50', '4.50', '6.50', '7.50']]
    res_most_cons_name = [['N', 'D', 'R', 'P', 'P'], ['P', 'R', 'W', 'P', 'P']]

    """first and last residue number of each TM"""
    TM_begin = [12, 48, 83, 128, 171, 292, 336]
    TM_end = [43, 77, 118, 153, 212, 329, 361]

    """Read in the ref_sequences_file as a file handle"""
    if not os.path.isfile(ref_sequences_file):
        raise IOError('Reference sequences file not found. Please specify the path with -ref')
    with open(ref_sequences_file, 'r') as f:
        ref_seq_fasta = f.read()
    input_handle = StringIO(ref_seq_fasta)

    """Perform both the alignment and calculation"""
    for rec in SeqIO.parse(input_handle, "fasta"):
        if rec.id == "hrh4-human":
            # Return the hH4R sequence
            hH4R_seq = ''.join([i for i in list(rec) if i != '-'])
            hH4R_sequence = get_res_num_name_pairs(hH4R_seq, 1)
            hH4R_sequence_dict = dict(hH4R_sequence)

            # Parse the sample .pdb/.ent/.cif structure, return the sample sequence and write it to the sample_sequence_file (in fasta format)
            sample_structure = parse_structure(del_res[0], del_res[1])
            sample_sequence = get_pdb_sequence(sample_structure)
            sample_sequence_dict = dict(sample_sequence)
            sample_seq = ''.join([i[1] for i in sample_sequence])
            with open(sample_sequence_file, 'w') as f:
                f.write(">sample" + '\n' + sample_seq)

            # Alignment: Return the mapping between hH4R and the sample sequence
            align_sequences()

            # Return both, residue/atom numbers/distances used for calculating the Universal Activation Index, and the Index itself
            get_ai_index()

            # Be verbose
            verbose(sample_sequence)

            break
