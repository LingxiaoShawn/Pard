import torch
import pickle
import os.path as osp
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import shutil, os 

class ARMDataset(InMemoryDataset):
    def __init__(self, dataset_name, root='data/ARM', split='train', transform=None, pre_transform=None):
        self.name = dataset_name
        self.split = split
        super().__init__(f'{root}/{dataset_name}', transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f"{self.name}.pkl"
    
    @property
    def processed_file_names(self):
        return f'{self.name}_{self.split}.pt' 
    
    def download(self):
        print('Copying files from source path ...')
        src = os.path.join(self.root, os.pardir, f"{self.name}.pkl")
        shutil.copyfile(src, self.raw_paths[0])

    def process(self):
        # Read data into huge `Data` list. 
        graphs = pickle.load(open(self.raw_paths[0], 'rb'))
        # transform graphs from nx to data list 
        data_list = [from_networkx(nx_graphs) for nx_graphs in graphs]
        if hasattr(data_list[0], 'label'):
            labels = [item for data in data_list for item in data.label ]
            node_encoding = {string: i for i, string in enumerate(set(labels))}
            edge_labels = [item for data in data_list for item in data.edge_label ]
            edge_encoding = {string: i for i, string in enumerate(set(edge_labels))}
            def _transform(data):
                data.x = torch.tensor([node_encoding[label] for label in data.label])
                data.edge_attr = torch.tensor([edge_encoding[label] for label in data.edge_label])
                del data.label, data.edge_label
                return data
            data_list = [_transform(data) for data in data_list]

        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        num_graphs = len(data_list)
        num_train = int(num_graphs * 0.8)
        num_val = int(num_train * 0.2)
        num_train = num_train - num_val
        
        lists = {
            'train': data_list[num_val:num_val+num_train],
            'val': data_list[:num_val],
            'test': data_list[num_train+num_val:]
        }
        for split in ['train', 'val', 'test']:
            torch.save(self.collate(lists[split]), osp.join(self.processed_dir, f'{self.name}_{split}.pt'))
