
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected
import numpy as np 
import random 
import os.path as osp

class SimulatedGraphs(InMemoryDataset):
    def __init__(self, type, root='data', split='train', transform=None, pre_transform=None, seed=15213):
        self.graph_type = type
        self.split = split
        assert split in ['train', 'val', 'test']
        self.random_seed = seed
        super().__init__(f'{root}/{type}', transform, pre_transform)
        self.data, self.slices, self.bandwidth = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["generated_data_train.g6", "generated_data_val.g6", "generated_data_test.g6", "bandwidth.txt"]

    @property
    def processed_file_names(self):
        return f'data_{self.split}.pt'

    def download(self):
        # Download to `self.raw_dir`.
        # Create graph and save them to raw_dir 
        print("Generating dataset...")
        graph_list, bandwidth = create_graphs(self.graph_type, self.random_seed)
        random.seed(0)
        random.shuffle(graph_list)

        num_graphs = len(graph_list)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))

        with open(self.raw_paths[0], 'wb') as f:
            for g in graph_list[:train_len]:
                f.write(nx.to_graph6_bytes(g))
        with open(self.raw_paths[2], 'wb') as f:
            for g in graph_list[train_len:train_len+test_len]:
                f.write(nx.to_graph6_bytes(g))  
        with open(self.raw_paths[1], 'wb') as f:
            for g in graph_list[train_len+test_len:]:
                f.write(nx.to_graph6_bytes(g))  

        with open(self.raw_paths[3], 'w') as f:
            f.write(f"{bandwidth:d}")

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(osp.join(self.raw_dir, f'generated_data_{self.split}.g6'))

        with open(self.raw_paths[-1], "r") as f:
            bandwidth = int(f.readline().strip())
        data_list = []
        for i,datum in enumerate(dataset):
            # x = torch.ones(datum.number_of_nodes(), 1, dtype=torch.int64) 
            # plain graph doesn't need x
            num_nodes = datum.number_of_nodes()
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))  
            x = torch.zeros(num_nodes, 1, dtype=torch.long)          
            data_list.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, bandwidth), self.processed_paths[0])

# Routine to create datasets 
# From GraphRNN code, https://github.com/snap-stanford/GraphRNN/blob/master/create_graphs.py
# Modified for newest NetworkX

def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)

    # nx.connected_components(G)

    G = G.subgraph(max(nx.connected_components(G), key=len))
    return G

def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    # list(nx.connected_component(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    return G

def create_graphs(graph_type, seed=15213):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    # base_path = os.path.join(dataset_path, f'{graph_type}/')
    graphs = []
    if graph_type=='ladder':
        default_bandwidth = 10
        for i in range(100, 201):
            graph = nx.ladder_graph(i)
            graphs.append(graph)
    elif graph_type=='ladder_small':
        default_bandwidth = 10
        for i in range(2, 11):
            graph = nx.ladder_graph(i)
            graphs.append(graph)
    elif graph_type=='tree':
        default_bandwidth = 25
        for i in range(2, 5):
            for j in range(3, 5):
                graph = nx.balanced_tree(i,j)
                graphs.append(graph)
    elif graph_type=='caveman':
        default_bandwidth = 100
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graph = caveman_special(i, j, p_edge=0.3)
                    graphs.append(graph)
    elif graph_type=='caveman_small':
        default_bandwidth = 20
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(100):
                    graph = caveman_special(i, j, p_edge=0.8) # default 0.8
                    graphs.append(graph)
    elif graph_type=='path':
        default_bandwidth = 50
        for l in range(2, 51):
            graph = nx.path_graph(l) # default 0.8
            graphs.append(graph)
    elif graph_type=='caveman_small_single':
        default_bandwidth = 20
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graph = caveman_special(i, j, p_edge=0.5)
                    graphs.append(graph)
    elif graph_type.startswith('community'):
        num_communities = int(graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        # c_sizes = [15] * num_communities
        for k in range(3000): # revise from 3000 to 500
            c_sizes = np.random.choice(np.arange(start=15,stop=30), num_communities)
            graph = n_community(c_sizes, p_inter=0.01)
            graphs.append(graph)
        default_bandwidth = 80
    elif graph_type=='grid':
        default_bandwidth = 40
        for i in range(10,20):
            for j in range(10,20):
                graph = nx.grid_2d_graph(i,j)
                graphs.append(graph)
    elif graph_type=='grid_small':
        default_bandwidth = 15
        for i in range(2,20):
            for j in range(2,8):
                graph = nx.grid_2d_graph(i, j)
                nodes = list(graph.nodes())
                node_mapping = {nodes[i]: i for i in range(len(nodes))}
                graph = nx.relabel_nodes(graph, node_mapping)
                graphs.append(graph)
    elif graph_type=='grid_big':
        default_bandwidth = 90
        for i in range(36, 46):
            for j in range(36, 46):
                graph = nx.grid_2d_graph(i,j)
                graphs.append(graph)
    elif graph_type=='barabasi':
        default_bandwidth = 130
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                     graph = nx.barabasi_albert_graph(i,j)
                     graphs.append(graph)
    elif graph_type=='barabasi_small':
        default_bandwidth = 20
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                     graph = nx.barabasi_albert_graph(i,j)
                     graphs.append(graph)
    else:
        print('Dataset - {} is not valid'.format(graph_type))
        exit()

    return graphs, default_bandwidth

if __name__ == "__main__":
    gs = create_graphs('community2')