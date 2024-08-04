"""
Adapted from DiGress. 
Based on GDSS code (modified).
https://github.com/harryjo97/GDSS
"""

import numpy as np
import networkx as nx
import torch
import re
import wandb

try:
    from rdkit import Chem
    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# ATOM_ENCODER = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'B': 5, 'Br': 6,
#                  'Cl': 7, 'I': 8, 'P': 9, 'S': 10, 'Se': 11, 'Si': 12}
# ATOM_DECODER = ['H', 'C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

class BasicMolecularMetrics:
    def __init__(self, atom_decoder, train_smiles=None, test_smiles=None):
        self.atom_decoder = atom_decoder
        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles
        self.test_smiles_list = None # there is a bug in nspdk computation, so we skip it for now. 
        self.remove_h = atom_decoder[0] != 'H' 

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.atom_decoder)
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)
    
    def compute_nspdk(self, mols):
        if self.test_smiles_list is None:
            print("Test dataset smiles is None, nspdk computation skipped")
            return 1
        test_mols = smiles_to_mols(self.test_smiles_list)
        test_nx = mols_to_nx(test_mols)
        mols = correct_mols(mols)
        generated_nx = mols_to_nx(mols)
        nspdk_mmd_dist = compute_nspdk_mmd(test_nx, generated_nx, metric='nspdk', is_hist=False, n_jobs=20)
        return nspdk_mmd_dist
        
    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        mols = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.atom_decoder, relax=True)
            smiles = mol2smiles(mol)
            if mol is not None:
                mols.append(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated), mols

    def evaluate(self, generated):
        """ generated: list of pairs (atom_types: n , edge_types: n x n) """
        valid, validity, num_components, all_smiles = self.compute_validity(generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        relaxed_valid, relaxed_validity, mols = self.compute_relaxed_validity(generated)
        print(f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []

        nspdk = self.compute_nspdk(mols)
        return ([validity, relaxed_validity, uniqueness, novelty, nspdk], 
                dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), unique, all_smiles)
    
    def __call__(self, molecule_list):
        """ molecule_list: (dict) """

        if not self.remove_h:
            print(f'Analyzing molecule stability...')

            molecule_stable = 0
            nr_stable_bonds = 0
            n_atoms = 0
            n_molecules = len(molecule_list)

            for i, mol in enumerate(molecule_list):
                atom_types, edge_types = mol

                validity_results = check_stability(atom_types, edge_types, atom_decoder=self.atom_decoder)

                molecule_stable += int(validity_results[0])
                nr_stable_bonds += int(validity_results[1])
                n_atoms += int(validity_results[2])

            # Validity
            fraction_mol_stable = molecule_stable / float(n_molecules)
            fraction_atm_stable = nr_stable_bonds / float(n_atoms)
            validity_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}
            if wandb.run:
                wandb.log(validity_dict)
        else:
            validity_dict = {'mol_stable': -1, 'atm_stable': -1}
        rdkit_metrics = self.evaluate(molecule_list)
        all_smiles = rdkit_metrics[-1]
        unique_smiles = rdkit_metrics[-2]
        nc = rdkit_metrics[1]
        dic = {'Validity': rdkit_metrics[0][0], 'Relaxed Validity': rdkit_metrics[0][1],
            'Uniqueness': rdkit_metrics[0][2], 'Novelty': rdkit_metrics[0][3], 'GDSS_NSPDK': rdkit_metrics[0][4],
            'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu']}
        if wandb.run:
            wandb.log(dic)

        return validity_dict, dic, unique_smiles, all_smiles

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def mols_to_smiles(mols):
    smiles = [mol2smiles(mol) for mol in mols]
    return [x for x in smiles if x is not None]

def smiles_to_mols(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [x for x in mols if x is not None]

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))      
        nx_graphs.append(G)
    return nx_graphs

def build_molecule(atom_types, edge_types, atom_decoder, verbose=False, relax=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types) #### here assume edge type start from 0, with 0 being not connected
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()] )
            if relax:
                flag, atomid_valence = check_valency(mol)
                if verbose:
                    print("flag, valence", flag, atomid_valence)
                if flag:
                    continue
                else:
                    assert len(atomid_valence) == 2
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if verbose:
                        print("atomic num of atom with a large valence", an)
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

def correct_mols(mols):
    cmols = []
    for mol in mols:
        cmol, no_correct = correct_mol(mol)
        vcmol = valid_mol_can_with_seg(cmol)
        if vcmol is not None:
            cmols.append(vcmol)
    return cmols

# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol

def check_stability(atom_types, edge_types, atom_decoder, debug=False):
    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(molecule_list, train_smiles, atom_decoder, remove_h=False):
    """ molecule_list: (dict) """

    if not remove_h:
        print(f'Analyzing molecule stability...')

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(atom_types, edge_types, atom_decoder=atom_decoder)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # stability_dict
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}
        if wandb.run:
            wandb.log(stability_dict)
    else:
        stability_dict = {'mol_stable': -1, 'atm_stable': -1}

    metrics = BasicMolecularMetrics(atom_decoder, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]
    if wandb.run:
        nc = rdkit_metrics[-2]
        dic = {'Validity': rdkit_metrics[0][0], 'Relaxed Validity': rdkit_metrics[0][1],
               'Uniqueness': rdkit_metrics[0][2], 'Novelty': rdkit_metrics[0][3],
               'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu']}
        wandb.log(dic)

    return stability_dict, rdkit_metrics, all_smiles



from sklearn.metrics.pairwise import pairwise_kernels
from .gdss_utils import vectorize

def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)