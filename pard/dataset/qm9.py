import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader

from pard.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from pard.analysis.rdkit_functions import compute_molecular_metrics
from pard.utils import to_dense



def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])

class QM9Dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root, split, remove_h: bool, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def atom_decoder(self):
        if self.remove_h:
            return ['C', 'N', 'O', 'F']
        else:
            return ['H', 'C', 'N', 'O', 'F']
    @property
    def bond_weights(self):
        return [0, 1, 2, 3]
    
    @property
    def atom_weights(self):
        return [12, 14, 16, 19] if self.remove_h else [1, 12, 14, 16, 19]
    
    @property
    def atom_valencies(self):
        return [4, 3, 2, 1] if self.remove_h else [1, 4, 3, 2, 1]

    @property
    def edge_start_from_zero(self):
        return False

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return f'{self.split}.csv'  #['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self):
        if self.remove_h:
            return f'proc_{self.split}_no_h.pt'
        else:
            return f'proc_{self.split}_h.pt'

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(osp.join(self.raw_dir, self.split_file_name)):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        target_df = pd.read_csv(osp.join(self.raw_dir, self.split_file_name), index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)

        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = edge_type

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = torch.tensor(type_idx, dtype=torch.long)
            y = torch.zeros((1,), dtype=torch.float)

            if self.remove_h:
                type_idx = torch.tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_smiles(self, eval=False):
        # assert self.split == 'train', "Only train split need saving smiles."
        train_dataloader = DataLoader(self, batch_size=64, shuffle=False, num_workers=12)

        datadir = self.root
        smiles_file_name = f'{self.split}_smiles_no_h.npy' if self.remove_h else f'{self.split}_smiles_h.npy'
        smiles_path = os.path.join(datadir, smiles_file_name)
        if os.path.exists(smiles_path):
            print("Dataset smiles were found.")
            train_smiles = np.load(smiles_path)
        else:
            print("Computing dataset smiles...")
            train_smiles = compute_qm9_smiles(self.atom_decoder, train_dataloader, self.remove_h)
            np.save(smiles_path, np.array(train_smiles))

        if eval:
            all_molecules = []
            for i, data in enumerate(train_dataloader):
                X, E, node_mask = to_dense(data, one_hot=False)
                for k in range(X.size(0)):
                    n = node_mask[k].sum()
                    atom_types = X[k, :n].cpu()
                    edge_types = E[k, :n, :n].cpu()
                    all_molecules.append([atom_types, edge_types])

            print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
            metrics = compute_molecular_metrics(all_molecules, train_smiles, self.atom_decoder, remove_h=self.remove_h)
            print(metrics[0])
        return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    num_mols = 0
    for i, data in enumerate(train_dataloader):
        X, E, node_mask = to_dense(data, one_hot=False)
        n_nodes = node_mask.sum(dim=-1)

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1
        num_mols += len(molecule_list)

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid, "percentage={0:.2%}".format(float(invalid) / num_mols))
    print("Number of disconnected molecules", disconnected, "percentage={0:.2%}".format(float(disconnected) / num_mols))
    return mols_smiles
