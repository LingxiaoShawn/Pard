"""
    from https://github.com/cvignac/DiGress/blob/main/src/datasets/moses_dataset.py
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
import os.path as osp
import os
import pathlib
from typing import Any, Sequence
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, download_url
from pard.analysis.rdkit_functions import mol2smiles, build_molecule
from pard.utils import to_dense

import pandas as pd


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'
    atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']

    def __init__(self, root, split, filtered: bool, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        assert split in ['train', 'val', 'test']
        self.filter_dataset = filtered
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def atom_decoder(self):
        return atom_decoder
    @property
    def bond_weights(self):
        return [0, 1, 2, 3, 1.5]
    
    @property
    def atom_weights(self):
        return [12, 14, 32, 16, 19, 35.4, 79.9, 1]
    
    @property 
    def atom_valencies(self):
        return [4, 3, 4, 2, 1, 1, 1, 1]
    
    @property
    def edge_start_from_zero(self):
        return False
    
    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_file_name(self):
        return f'{self.split}_moses.csv'

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return f'{self.split}_filtered.pt'
        else:
            return f'{self.split}.pt'

    def download(self):
        import rdkit  # noqa
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))
        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'val_moses.csv'))
        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'test_moses.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        path = osp.join(self.raw_dir, self.split_file_name)
        smiles_list = pd.read_csv(path)['SMILES'].values

        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smiles_list)):
            mol = Chem.MolFromSmiles(smile)
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

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = edge_type
 
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            x = torch.tensor(type_idx, dtype=torch.long)
            y = torch.zeros(size=(1,), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                X, E, node_mask = to_dense(data, one_hot=False)

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule(atom_types, edge_types, atom_decoder, relax=True)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        if self.filter_dataset:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, f'new_{self.split}.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")