import torch
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.data import Data, Batch

from .utils import get_causal_block_mask
import torch.nn.functional as F
from torch_geometric.data.collate import collate

def compute_higher_order_degrees(edge_index, num_nodes, num_hops=2):
    """
    TODO: can even support subgraph level check, by using AI, where A is sparse matrix of edge_index, I is identity matrix.
    1. for each node, get the distance-to-this-node vector, then sum over all weighted distances. By d * (num_nodes**(max_degree - d))
    """
    x = torch.ones(num_nodes, dtype=torch.long, device=edge_index.device)
    weighted_degrees = 0 
    for _ in range(num_hops):
        scatter(x[edge_index[0]], edge_index[1], dim=0, reduce="sum", out=x)
        weighted_degrees = weighted_degrees * num_nodes + x # times num_nodes to make sure that check lower order degree first
    return weighted_degrees

class ToParallelBlocks:
    # Requires input graphs are **one-hot encoded**
    def __init__(self, max_hops=2, add_virtual_blocks=True, random_partial_blocks=False, to_batched_sequential=False):
        self.max_hops = max_hops
        self.add_virtual_blocks = add_virtual_blocks
        self.random_partial_blocks = random_partial_blocks
        self.to_sequential_list = to_batched_sequential # this avoid save all graphs as a new dataset, but this increase the true number of batch size
        self.add_virtual_edges = False

    def __call__(self, graph):
        has_attr = True if hasattr(graph, 'edge_attr') else None # True or None

        # Get block id based on weighted degree
        nodes = torch.arange(graph.num_nodes, dtype=int)
        block_id = torch.zeros(graph.num_nodes, dtype=int)
        block_degree, block_size, i = [], [], 1
        edge_index = graph.edge_index
        while block_id.min() == 0:
            current_degree = degree(edge_index[0], graph.num_nodes, dtype=torch.long)
            weighted_degrees = compute_higher_order_degrees(edge_index, graph.num_nodes, num_hops=self.max_hops)
            weighted_degrees[block_id!=0] = -1
            min_degree = weighted_degrees[weighted_degrees>=0].min()
            block_nodes = nodes[weighted_degrees==min_degree] 
            block_id[block_nodes] = i
            # update the graph, by deleting the block 
            block_mapping = (block_id == i)
            edge_index = edge_index[:, ~(block_mapping[edge_index].any(0))]
            block_degree.append(current_degree[block_nodes][0]) # all nodes in the block share the same degree
            block_size.append(block_nodes.size(0))
            i += 1

        # Reverse the order of blocks
        block_id = block_id.max() - block_id # start from 0
        block_degree = torch.tensor(block_degree[::-1]).long() # (num_blocks)
        block_size = torch.tensor(block_size[::-1]).long()     # (num_blocks)
        # Sort blocks nodes
        block_id, sorted_nodes = block_id.sort()
        node_mapping = nodes.clone()
        node_mapping[sorted_nodes] = nodes

        x = graph.x[sorted_nodes]                       # sort nodes
        edge_index = node_mapping[graph.edge_index]     # relabel edges
        edge_attr = graph.edge_attr if has_attr else None
        is_virtual_node = x[:, -1].bool() 
        
        if self.add_virtual_blocks:
            assert not self.random_partial_blocks, 'add_virtual_blocks cannot be used with random_partial_blocks'
            assert not self.to_sequential_list, 'add_virtual_blocks cannot be used with to_sequential_list'
            x = F.pad(x, (0, 0, 0, graph.num_nodes)) # pad one virtual node at the end
            is_virtual_node = F.pad(is_virtual_node, (0, graph.num_nodes), value=True) # pad one virtual node at the end
            block_id = torch.cat([block_id, block_id], dim=0)
            if self.add_virtual_edges:
                # Add virtual edges (this may not necessary as we don't use this as input)
                x[graph.num_nodes:, -1] = 1 # set the virtual node to 1
                causal_block = get_causal_block_mask(block_id, is_virtual_node)
                causal_block = causal_block | causal_block.t() # symmetric
                causal_block[torch.arange(causal_block.size(0)), torch.arange(causal_block.size(0))] = False # remove self-loop
                causal_block[:graph.num_nodes, :graph.num_nodes] = False # remove edges between real nodes
                virtual_edges = causal_block.nonzero(as_tuple=False).t() # get all edges from the causal block
                edge_index = torch.cat([edge_index, virtual_edges], dim=1)
                if has_attr:
                    virtual_edges_attr = torch.zeros(virtual_edges.size(1), graph.edge_attr.size(1))
                    edge_attr = torch.cat([graph.edge_attr, virtual_edges_attr], dim=0)

        num_blocks = block_id.max()+1
        def get_until_ith_id(l):
            node_mask = (block_id<=l) # N
            edge_mask = (edge_index<node_mask.sum()).all(0)
            return Data(
                x=x[node_mask],
                edge_index=edge_index[:, edge_mask],                              # 2 x E
                edge_attr=edge_attr[edge_mask] if has_attr else None,             # E x C    
                node_block_id=block_id[node_mask],                                # N
                is_virtual_node=is_virtual_node[node_mask],                       # N
                block_degree=block_degree[:l+1],                # (num_real_blocks) doesn't consider virtual blocks 
                block_size=block_size[:l+1],                    # (num_real_blocks) 
                block_batch=torch.zeros(l+1, dtype=torch.long), # (num_real_blocks), this is added for split the sparse batch to dense batch
                next_block_size=torch.tensor([block_size[l+1]]) if l+1<num_blocks else torch.tensor([0], dtype=torch.long),
                next_block_degree=torch.tensor([block_degree[l+1]]) if l+1<num_blocks else  torch.tensor([0], dtype=torch.long),
            )
        if self.to_sequential_list:
            # here we batch all partial graphs together
            graph_list = [get_until_ith_id(l) for l in range(num_blocks)]
            batched_graphs = collate(Data, graph_list)[0]
            batched_graphs.graph_batch = batched_graphs.batch.clone() # the original batch will be erased by loader 
            return batched_graphs
        if self.random_partial_blocks:
            l = torch.randint(num_blocks, (1,)).item()
            return get_until_ith_id(l)
        return get_until_ith_id(num_blocks-1)
    
class ToDenseParallelBlocksBatch:
    def __init__(self, one_hot=True):
        self.one_hot = one_hot
    def __call__(self, batch):
        # inputs 
        x            = batch.x
        edge_index   = batch.edge_index
        edge_attr    = batch.edge_attr
        block_id     = batch.node_block_id
        block_degree = batch.block_degree
        block_size   = batch.block_size
        block_batch  = batch.block_batch
        graph_batch  = batch.batch if not hasattr(batch, 'graph_batch') else batch.graph_batch

        is_virtual_node = batch.is_virtual_node if hasattr(batch, 'is_virtual_node') else False
        # stats
        batch_size = int(graph_batch.max()) + 1
        num_nodes  = scatter_add(torch.ones_like(graph_batch), graph_batch, dim=0, dim_size=batch_size)
        num_blocks = scatter_add(torch.ones_like(block_batch), block_batch, dim=0, dim_size=batch_size)
        max_num_nodes, max_num_blocks = num_nodes.max(), num_blocks.max() 
        # create masks 
        node_mask  = torch.arange(max_num_nodes, device=x.device).unsqueeze(0) < num_nodes.unsqueeze(-1)
        block_mask = torch.arange(max_num_blocks, device=x.device).unsqueeze(0) < num_blocks.unsqueeze(-1) 

        # create dense node features
        dense_x  = torch.zeros(batch_size, max_num_nodes, x.size(-1), device=x.device)
        dense_id = -torch.ones(batch_size, max_num_nodes, dtype=torch.long, device=x.device) # default -1
        dense_virtual = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool, device=x.device)
        dense_x[node_mask]  = x
        dense_id[node_mask] = block_id
        dense_virtual[node_mask] = is_virtual_node 
        # create dense block features
        dense_degree = torch.zeros(batch_size, max_num_blocks, dtype=torch.long, device=x.device)
        dense_size   = torch.zeros(batch_size, max_num_blocks, dtype=torch.long, device=x.device)
        dense_degree[block_mask] = block_degree
        dense_size[block_mask]   = block_size
        # create adj matrix 
        adj = to_dense_adj(edge_index, graph_batch, edge_attr, max_num_nodes=max_num_nodes)
        if self.one_hot:
            adj[:,:,:, 0] = 1 - adj.sum(dim=-1)                         # encode the disconnected edge to 0-class
            adj[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = 0 # mask out the mask part 
            adj[:, torch.eye(adj.size(1), dtype=torch.bool)] = 0        # mask out the diagonal part
        else:
            adj[:, torch.eye(adj.size(1), dtype=torch.bool)] = 0        # set diagonal part disconnected
            adj[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = -1 # mask out the padding part 

        # create new dense batch
        return Batch(
            edges=adj,                          # B x Nmax x Nmax x C
            nodes=dense_x,                      # B x Nmax x F
            nodes_blockid=dense_id,             # B x Nmax,           <0 part is padding => node_mask 
            virtual_node_mask=dense_virtual,    # B x Nmax
            block_degree=dense_degree,          # B x max_real_block, contains 0 for the first block
            block_size=dense_size,              # B x max_real_block, 0 is padding => block_mask
            next_block_size=batch.next_block_size if hasattr(batch, 'next_block_size') else None,     # B x 1
            next_block_degree=batch.next_block_degree if hasattr(batch, 'next_block_degree') else None, # B x 1
            to_original_graphs=scatter_max(batch.batch, batch.graph_batch, dim=0, dim_size=batch_size)[0] if hasattr(batch, 'graph_batch') else None, # B
        )