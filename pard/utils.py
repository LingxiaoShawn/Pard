import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch

from torch_geometric.utils import to_dense_adj, to_dense_batch
def to_dense(batch, one_hot=True):
    """
    This function removes self-loop 
    """
    batch_idx = None
    if hasattr(batch, 'batch'):
        batch_idx = batch.batch

    x, mask = to_dense_batch(batch.x, batch_idx)
    adj = to_dense_adj(batch.edge_index, batch_idx, batch.edge_attr, max_num_nodes=x.size(1))

    if one_hot:
        # encode the disconnected edge to 0-class
        adj[:,:,:, 0] = 1 - adj.sum(dim=-1)
        # mask out the mask part 
        adj[~(mask.unsqueeze(1) * mask.unsqueeze(2))] = 0
        # mask out the diagonal part
        adj[:, torch.eye(adj.size(1), dtype=torch.bool)] = 0
    else:
        # set diagonal part disconnected
        adj[:, torch.eye(adj.size(1), dtype=torch.bool)] = 0
        # mask out the padding part 
        adj[~(mask.unsqueeze(1) * mask.unsqueeze(2))] = -1
    return x, adj, mask

def plot_nx_graphs(G_list, node_size=20, edge_size=0.2, layout='spring', node_attr='x', edge_attr='edge_attr', block=None):
    N = len(G_list)
    n_rows = int(np.sqrt(N))
    n_cols = int(np.ceil(N/n_rows))
    plt.figure(figsize=(n_rows*4, n_cols*3))
    for i, G in enumerate(G_list):
        if layout=='spring':
            node_pos = nx.spring_layout(G)
        else:
            node_pos = nx.spectral_layout(G) # use spectral layout
        plt.subplot(n_rows, n_cols, i+1)
        node_color = 'blue'
        edge_color = 'k'
        # [edge[-1] for edge in nx_list[0].edges(data='edge_attr')]
        if node_attr is not None:
            node_color =  [node[-1] for node in G.nodes(data=node_attr)]
            # nx.get_node_attributes(G, node_attr)
        if edge_attr is not None:
            edge_color = [edge[-1] for edge in G.edges(data=edge_attr)]
        final_node_size = node_size
        if block is not None:
            block_indicator = np.array([node[-1] for node in G.nodes(data=block)])
            final_node_size = node_size * (1 + 5*block_indicator) # enlarge the block nodes 5 tmes
        nx.draw(G, node_size=final_node_size, node_color=node_color, edge_color=edge_color,
                   linewidths=edge_size, pos=node_pos,  cmap='coolwarm')
    return plt

def plot_pgy_graphs(data_list, node_size=20, edge_size=0.2, layout='spring', node_attr='x', edge_attr='edge_attr', block=None):  
    G_list = [to_networkx(data, to_undirected='lower', node_attrs=([node_attr] if node_attr else []) + ([block] if block else []), 
                                edge_attrs= edge_attr and [edge_attr]) for data in data_list]
    return plot_nx_graphs(G_list, node_size, edge_size, layout, node_attr, edge_attr, block)


def from_batch_onehot_to_pyg_list(nodes, edges, blocks=None):
    # step 1: extract node features 
    node_mask = nodes.sum(-1) == 1
    batch_idx, node_idx = node_mask.nonzero(as_tuple=True)
    x = nodes[batch_idx, node_idx]

    # step 2: extract edge index
    edge_mask = edges.sum(-1) == 1
    edge_mask = edge_mask & (edges[:,:,:,0]==0) # 0th position being 0 indicate connection
    edge_batch, edge_index_left, edge_index_right = edge_mask.nonzero(as_tuple=True)
    # create graph attri
    edge_attr = edges[edge_batch, edge_index_left, edge_index_right] #[:,1:]

    if blocks is not None:
        notpadding = (blocks >= 0) # B x Nmax
        block_row_idx, block_col_idx = notpadding.nonzero(as_tuple=True)
        block_id = blocks[block_row_idx, block_col_idx]

    data_list = []
    for i in range(nodes.size(0)):
        x_i = x[batch_idx==i]
        edge_idx = edge_batch==i
        edge_index_i = torch.stack([edge_index_left[edge_idx], edge_index_right[edge_idx]])
        edge_attr_i = edge_attr[edge_idx]
        # transform x and edge_attr from one-hot to integer for networkx
        x_i = x_i.argmax(-1)
        edge_attr_i = edge_attr_i.argmax(-1) # here 0 is disconnected, hence edge label starting from always starting from 1. This is align with ToOneHot
        block_id_i = block_id[block_row_idx==i] if blocks is not None else None
        data_list.append(Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i, block_id=block_id_i))
        
    return data_list

from pard.parallel.transform import ToParallelBlocks
from functools import partial
def check_block_id_train_vs_generation(generated_nodes, generated_edges, generated_blocks, train_max_hops=3):
    data_list = from_batch_onehot_to_pyg_list(generated_nodes, generated_edges, generated_blocks)
    transform = ToParallelBlocks(max_hops=train_max_hops)
    transform = partial(transform, only_return_blockid=True)
    data_list = [transform(data) for data in data_list]
    equal = [(g.block_id == g.node_block_id).all().int().item() for g in data_list]
    return equal

def from_batch_onehot_to_list(nodes, edges):
    # this is the format needed by metrics from DiGress
    node_mask = nodes.sum(-1) == 1 # B x N 
    num_nodes = node_mask.sum(-1)
    return [(nodes[i][:num_nodes[i]].argmax(-1) , edges[i][:num_nodes[i], :num_nodes[i]].argmax(-1)) for i in range(len(num_nodes))] 
     

import os, re
def find_checkpoint_with_lowest_val_loss(directory, return_latest=False):
    lowest_val_loss = float('inf')
    largest_epoch = 0
    best_checkpoint = None
    latest_checkpoint = None
    # Regex to match the pattern in the filename and extract the loss value
    pattern = re.compile(r'epoch=([\d.]+)+-val_loss=([\d.]+)\.ckpt')

    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            val_loss = float(match.group(2))
            epoch = int(match.group(1))
            if epoch > largest_epoch:
                largest_epoch = epoch
                latest_checkpoint = filename
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_checkpoint = filename
    if return_latest and latest_checkpoint is not None:
        return os.path.join(directory, latest_checkpoint)
    elif best_checkpoint is not None:
        return os.path.join(directory, best_checkpoint)
    else:
        return "No matching checkpoint found."