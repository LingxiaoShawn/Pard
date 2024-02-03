import torch, json
import numpy as np 
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url

from .mol_preprocess import GGNNPreprocessor, DataFrameParser
class ZINC250k(InMemoryDataset):
    raw_data_url = 'https://raw.githubusercontent.com/DSL-Lab/SwinGNN/main/dataset/zinc250k.csv'
    raw_idx_url = 'https://raw.githubusercontent.com/DSL-Lab/SwinGNN/main/dataset/valid_idx_zinc250k.json'
    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'test', 'val']
        if split == 'val':
            split = 'test' ## no validation set, use test set directly, which is the same as previous work
        self.split = split
        super().__init__(f'{root}/zinc250k', transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def atom_decoder(self):
        return ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'] # 9 different types
    
    @property
    def raw_file_names(self):
        return ['zinc250k.csv', 'valid_idx_zinc250k.json']
    
    @property
    def processed_file_names(self):
        return [f'data_{self.split}.pt', f'smiles_{self.split}.npy']
    
    def download(self):
        download_url(self.raw_data_url, self.raw_dir)
        download_url(self.raw_idx_url, self.raw_dir)

    def process(self):
        ### read graphs from csv 
        max_atoms = 38
        label_idx = 1
        smiles_col = 'smiles'
        preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
        df = pd.read_csv(self.raw_paths[0], index_col=0) # csv of smiles 
        # Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
        # 'smiles' column contains '\n', need to remove it.
        # Here we do not remove \n, because it represents atom N with single bond
        labels = df.keys().tolist()[label_idx:]
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
        result = parser.parse(df, return_smiles=True)

        # get smile strings
        smiles = result['smiles']   # array of smile strings\
        # convert graphs to Data
        dataset = result['dataset'] # NumpyTupleDataset, list of (X,ADJ,y_label_of_mol)
        
        # build atomic number encoder
        # AtomNumber_TO_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0] # 0 for padding virtual node
        atom_mapper = -torch.ones(54, dtype=torch.long)
        atom_mapper[atomic_num_list] = torch.arange(len(atomic_num_list)) # # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        data_list = []
        for data in dataset:
            x, adj, y = data
            ### remove padding 
            num_nodes = (x != 0).sum()
            x = x[:num_nodes]
            adj = adj[...,:num_nodes, :num_nodes]
            ### transform atom number to idx. 
            x = torch.from_numpy(x).long()
            x = atom_mapper[x] 
            ### encode adj, 4 x N x N, {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
            ### all previous work view BT.AROMATIC as disconnection 
            adj[-1,:,:] = 0 # remove BT.AROMATIC
            adj = torch.from_numpy(adj) 
            edge_attr, idx_1, idx_2 = adj.nonzero(as_tuple=True)
            edge_index = torch.stack([idx_1, idx_2], dim=0)  
            edge_attr = edge_attr.long() + 1 # start from 1

            data = Data(x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=torch.from_numpy(y).float())    
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data) 
            data_list.append(data)
            
        ### read test indice from json 
        all_idx = np.arange(len(data_list))
        with open(self.raw_paths[1], 'r') as f:
            test_idx = np.array(json.load(f)) # list of idx 
        train_idx = np.setdiff1d(all_idx, test_idx)
        indices = {'train': train_idx, 'test': test_idx}

        original_split = self.split 
        for split in ['train', 'test']:
            self.split = split
            train_data = [data_list[i] for i in indices[split]]
            train_smiles = smiles[indices[split]]
            np.save(self.processed_paths[1], np.array(train_smiles))
            torch.save(self.collate(train_data), self.processed_paths[0])
        self.split = original_split

    def get_smiles(self, eval=False):
        return np.load(self.processed_paths[1])