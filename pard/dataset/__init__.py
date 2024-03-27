from .simulated_graph import SimulatedGraphs
from .guacamol import GuacamolDataset
from .moses import MOSESDataset
from .qm9 import QM9Dataset
from .spectre import SpectreGraphDataset
from .arm_graph import ARMDataset
from.zinc250k import ZINC250k

from torch_geometric.datasets import ZINC

from pard.analysis.spectre_utils import EMDSamplingMetrics, SBMSamplingMetrics, PlanarSamplingMetrics, MMDSamplingMetrics



DATA_INFO = {
    'zinc250k':{
        'class': ZINC250k,
        'num_node_features': 9,   ## compute by dataset.data.x.max() + 1 (21 for subset=True, 28 for subset=False)
        'num_edge_features': 4,    ## compute by dataset.data.edge_attr.max() + 1
        'start_edge_type': 1,      ## compute by dataset.data.edge_attr.min()
        'default_args': {'root': 'data', 'split': 'train'},
        'atom_decoder': ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        'num_eval_generation': 10000,
    },
    'zinc':{
        'class': ZINC,
        'num_node_features': 21,   ## compute by dataset.data.x.max() + 1 (21 for subset=True, 28 for subset=False)
        'num_edge_features': 4,    ## compute by dataset.data.edge_attr.max() + 1
        'start_edge_type': 1,      ## compute by dataset.data.edge_attr.min()
        'default_args': {'root': 'data/ZINC', 'subset': True, 'split': 'train'}
    },
    'qm9':{
        'class': QM9Dataset,
        'num_node_features': 5,    ## compute by dataset.data.x.max() + 1
        'num_edge_features': 4,    ## compute by dataset.data.edge_attr.max() + 1
        'start_edge_type': 1,      ## compute by dataset.data.edge_attr.min()
        'default_args': {'root': 'data/QM9', 'split': 'train', 'remove_h': False},
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
        'num_eval_generation': 10000,
    }, 
    'moses':{
        'class': MOSESDataset,
        'num_node_features': 7,    
        'num_edge_features': 5,   
        'start_edge_type': 1,
        'default_args': {'root': 'data/moses', 'split': 'train', 'filtered': False},
        'atom_decoder': ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'],
        'num_eval_generation': 25000,
    },
    'guacamol':{
        'class': GuacamolDataset,
        'num_node_features': 12,    
        'num_edge_features': 5,    
        'start_edge_type': 1,
        'default_args': {'root': 'data/guacamol', 'split': 'train', 'filtered': False},
        'atom_decoder': ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si'],
    }, 
    'grid':{
        'class': SimulatedGraphs,
        'num_node_features': 1,  
        'num_edge_features': 0,
        'start_edge_type': 0,     
        'default_args': {'type': 'grid', 'root': 'data'},
        'metric_class': MMDSamplingMetrics,
        'num_eval_generation': 64,
    },
    'planar':{
        'class': SpectreGraphDataset,
        'num_node_features': 1,   
        'num_edge_features': 1,
        'start_edge_type': 0,    
        'default_args': {'dataset_name': 'planar', 'root': 'data', 'split': 'train'},
        'metric_class': PlanarSamplingMetrics, 
        'num_eval_generation': 40,
    },
    'sbm':{
        'class': SpectreGraphDataset,
        'num_node_features': 1,   
        'num_edge_features': 1,
        'start_edge_type': 0,    
        'default_args': {'dataset_name': 'sbm', 'root': 'data', 'split': 'train'},
        'metric_class': SBMSamplingMetrics,
        'num_eval_generation': 40,
    },
    'comm20':{
        'class': SpectreGraphDataset,
        'num_node_features': 1,   
        'num_edge_features': 1,
        'start_edge_type': 0,    
        'default_args': {'dataset_name': 'comm20', 'root': 'data', 'split': 'train'},
        'metric_class': EMDSamplingMetrics,
        'num_eval_generation': 40,
    },
    'caveman':{
        'class': ARMDataset,
        'num_node_features': 1,   
        'num_edge_features': 1,
        'start_edge_type': 0,   
        'default_args': {'dataset_name': 'caveman', 'root': 'data/ARM', 'split': 'train'},
        'metric_class': MMDSamplingMetrics, 
    },
    'breast':{
        'class': ARMDataset,
        'num_node_features': 10,   
        'num_edge_features': 3,
        'start_edge_type': 0,   
        'default_args': {'dataset_name': 'breast', 'root': 'data/ARM', 'split': 'train'},
        'metric_class': MMDSamplingMetrics, 
    },
    'cora':{
        'class': ARMDataset,
        'num_node_features': 7,   
        'num_edge_features': 1,
        'start_edge_type': 0,   
        'default_args': {'dataset_name': 'cora', 'root': 'data/ARM', 'split': 'train'},
        'metric_class': MMDSamplingMetrics, 
    },
    'ego_small':{
        'class': ARMDataset,
        'num_node_features': 0,   
        'num_edge_features': 0,
        'start_edge_type': 0,   
        'default_args': {'dataset_name': 'ego_small', 'root': 'data/ARM', 'split': 'train'},
        'metric_class': MMDSamplingMetrics, 
    },
    'community_small':{
        'class': ARMDataset,
        'num_node_features': 0,   
        'num_edge_features': 0,
        'start_edge_type': 0,   
        'default_args': {'dataset_name': 'community_small', 'root': 'data/ARM', 'split': 'train'},
        'metric_class': MMDSamplingMetrics, 
    },
}