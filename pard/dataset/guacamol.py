"""
    from https://github.com/cvignac/DiGress/blob/main/src/datasets/guacamol_dataset.py
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
import os
import os.path as osp
import hashlib
from typing import Any, Sequence
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, download_url


TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True

class GuacamolDataset(InMemoryDataset):
    train_url = ('https://figshare.com/ndownloader/files/13612760')
    test_url = 'https://figshare.com/ndownloader/files/13612757'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'
    all_url = 'https://figshare.com/ndownloader/files/13612745'

    atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

    def __init__(self, root, split, filtered: bool, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.filtered = filtered
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def atom_decoder(self):
        return ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

    @property
    def bond_weights(self):
        return [0, 1, 2, 3, 1.5]
    
    @property
    def atom_weights(self):
        return [12, 14, 16, 19, 10.81, 79.9, 35.45, 126.9, 30.97, 30.07, 78.97, 28.09]
    
    @property
    def atom_valencies(self):
        return [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]

    @property
    def edge_start_from_zero(self):
        return False
    
    @property
    def raw_file_names(self):
        if self.filtered:
            return ['new_train.smiles', 'new_val.smiles', 'new_test.smiles']
        else:
            return ['guacamol_v1_train.smiles', 'guacamol_v1_val.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_file_name(self):
        if self.filtered:
            return f'new_{self.split}.smiles'
        else:
            return f'guacamol_v1_{self.split}.smiles'
        
    @property
    def processed_file_names(self):
        if self.filtered:
            return f'new_proc_{self.split}.pt'
        else:
            return f'old_proc_{self.split}.pt'

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            train_path = download_url(self.train_url, self.raw_dir)
            os.rename(train_path, osp.join(self.raw_dir, 'guacamol_v1_train.smiles'))
            train_path = osp.join(self.raw_dir, 'guacamol_v1_train.smiles')

            test_path = download_url(self.test_url, self.raw_dir)
            os.rename(test_path, osp.join(self.raw_dir, 'guacamol_v1_test.smiles'))
            test_path = osp.join(self.raw_dir, 'guacamol_v1_test.smiles')

            valid_path = download_url(self.valid_url, self.raw_dir)
            os.rename(valid_path, osp.join(self.raw_dir, 'guacamol_v1_val.smiles'))
            valid_path = osp.join(self.raw_dir, 'guacamol_v1_val.smiles')
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)

        # check the hashes
        # Check whether the md5-hashes of the generated smiles files match
        # the precomputed hashes, this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit('Invalid hashes for the dataset files')

        print('Dataset download successful. Hashes are correct.')


    def process(self):

        RDLogger.DisableLog('rdApp.*')
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        smile_list = open(osp.join(self.raw_dir, self.split_file_name)).readlines()

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smile_list)):
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
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]                  #### edge start from 1, not 0

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = edge_type 

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = torch.tensor(type_idx, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=i)
        
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
