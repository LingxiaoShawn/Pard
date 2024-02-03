import torch 
from torch_geometric.utils import is_undirected

def get_node_edge_marginal_distribution(dataset):
    # this assume one-hot encoding, and only compute the real blocks
    node_dist = 0
    edge_dist = 0 
    for data in dataset:
        is_virtual_node = data.is_virtual_node
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = data.x 
        if hasattr(data, 'graph_batch'):
            # add support for batched sequential partial graphs
            last_graph_idx = (data.graph_batch == data.graph_batch.max())
            start_idx = (~last_graph_idx).sum()
            is_virtual_node = is_virtual_node[start_idx:]
            x = x[start_idx:]
            edge_mask = (edge_index >= start_idx).all(dim=0)
            edge_index = edge_index[:, edge_mask] - start_idx
            edge_attr = edge_attr[edge_mask]

        real_node_mask = ~is_virtual_node
        num_nodes = real_node_mask.sum()
        real_edge_mask = real_node_mask[edge_index].all(dim=0)
        num_edges = real_edge_mask.sum()
        assert is_undirected(edge_index)
        node_dist += x[real_node_mask].sum(dim=0)
        edge_dist += edge_attr[real_edge_mask].sum(dim=0)
        num_disconnected_edges = num_nodes * (num_nodes - 1) - num_edges
        edge_dist[0] += num_disconnected_edges

    edge_dist = edge_dist / edge_dist.sum()
    node_dist = node_dist / node_dist.sum()
    return node_dist, edge_dist

def get_init_block_size_degree_marginal_distrbution(dataset):
    # can be used for both with or without virtual block 
    assert hasattr(dataset[0], 'block_size') and hasattr(dataset[0], 'block_degree')
    max_block_size = max_block_degree = max_num_blocks = 0 
    # assume max_block_size and max_block_degree are smaller than 200
    init_size_freq = torch.zeros(200) 
    init_degree_freq = torch.zeros(200)
    for data in dataset:
        block_size = data.block_size
        block_degree = data.block_degree
        if hasattr(data, 'graph_batch'):
            # add support for batched sequential partial graphs
            last_graph_idx = (data.block_batch == data.block_batch.max())
            block_size = block_size[last_graph_idx]
            block_degree = block_degree[last_graph_idx]
        max_num_blocks = max(max_num_blocks, block_size.size(0))
        max_block_size = max(max_block_size, block_size.max())
        max_block_degree = max(max_block_degree, block_degree.max())
        init_size_freq[block_size[0]] += 1  # only record the marginal distribution of the first block 
        init_degree_freq[block_degree[0]] += 1
    max_block_size += 1 # include the 0 case
    max_block_degree += 1 # include the 0 case
    init_size_dist = init_size_freq[:max_block_size] / init_size_freq.sum()
    init_degree_dist = init_degree_freq[:max_block_degree] / init_degree_freq.sum()
    return init_size_dist, init_degree_dist, max_num_blocks
    
def get_last_target_block_mask(block_id, virtual_node_mask, lower_triangular_only=False, causal=True):
    # batched_block: B x N_max or  N_max
    N_max = block_id.size(-1)
    padding_mask = block_id == -1                                          # B x N_max
    node_mask = (block_id == block_id.max(dim=-1, keepdim=True)[0])        # B x N_max
    if causal:
        # in causal case, target block must be virtual block
        node_mask = node_mask & virtual_node_mask                              # B x N_max, only consider virtual node as target_block!!

    adj_mask = node_mask.unsqueeze(-2) & ~padding_mask.unsqueeze(-1)
    adj_mask = adj_mask | adj_mask.transpose(-1, -2)
    adj_mask[..., range(N_max), range(N_max)] = False
    if lower_triangular_only:
        idx1, idx2 = torch.triu_indices(N_max, N_max, offset=1)
        adj_mask[..., idx1, idx2] = False
    return node_mask, adj_mask

def get_target_block_mask(block_id, virtual_mask):
    causal_mask = get_causal_block_mask(block_id, virtual_mask) # B x Nmax x Nmax
    real_edge_mask, virtual_edge_mask = get_real_virtual_edge_mask(causal_mask, virtual_mask)
    node_mask = get_node_mask_from_block_id(block_id) # B x Nmax
    return virtual_mask, virtual_edge_mask, node_mask & (~virtual_mask), real_edge_mask

def get_causal_block_mask(block_id, nodes_virtual_mask=None, separate_diag_offdiag=False):
    if nodes_virtual_mask is None:
        nodes_virtual_mask = torch.zeros_like(block_id)
    assert block_id.shape == nodes_virtual_mask.shape # BxN or N
    left, right = block_id.unsqueeze(-1), block_id.unsqueeze(-2)
    edge_mask = (left >= 0) & (right >= 0)
    left_v, right_v = nodes_virtual_mask.unsqueeze(-1), nodes_virtual_mask.unsqueeze(-2)
    diagonal_mask = (left == right) & (left_v == right_v) & edge_mask
    off_diagonal_mask = (left > right) & (~right_v) & edge_mask
    if separate_diag_offdiag:
        return diagonal_mask, off_diagonal_mask
    asymmetric_causal_block_mask = (diagonal_mask | off_diagonal_mask)
    return asymmetric_causal_block_mask # BxNxN or NxN

def get_node_mask_from_block_id(nodes_block_id):
    return nodes_block_id >= 0 # B x Nmax

def get_edge_mask_from_block_id(nodes_block_id, nodes_virtual_mask=None):
    if nodes_virtual_mask is None:
        node_mask = get_node_mask_from_block_id(nodes_block_id)
        return node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # B x Nmax x Nmax
    else:
        causal_mask = get_causal_block_mask(nodes_block_id, nodes_virtual_mask)
        return causal_mask | causal_mask.transpose(-1, -2)     # B x Nmax x Nmax

def spread_blockproperty_to_nodes(block_property, nodes_block_id):
    """
    block_property: B x max_num_real_block x ... 
    nodes_block_id: B x Nmax

    return: B x Nmax x ...
    """
    batch_size = nodes_block_id.shape[0]
    device = nodes_block_id.device 
    node_mask = get_node_mask_from_block_id(nodes_block_id)

    idx = nodes_block_id.clone() 
    idx[~node_mask] = 0 # the original -1 cannot be used as index

    out = block_property[torch.arange(batch_size, device=device).unsqueeze(-1), nodes_block_id] # B x Nmax x ...
    out[~node_mask] = 0 # padding part
    return out 

from torch_scatter import scatter_add
def edge_to_node(x, block_id, virtual_mask, causal=True):
    causal_mask = get_causal_block_mask(block_id, virtual_mask) # B x Nmax x Nmax
    if not causal:
        causal_mask = causal_mask | causal_mask.transpose(-1, -2) # B x Nmax x Nmax
    node_mask = get_node_mask_from_block_id(block_id) # B x Nmax

    # aggregate from edges to nodes 
    num_elements = causal_mask.sum(dim=-1, keepdim=True).float() # B x Nmax x 1

    node_diag = torch.diagonal(x, dim1=1, dim2=2).transpose(-1,-2) # B x Nmax x C 
    node_offdiag = (x * causal_mask.unsqueeze(-1)).sum(dim=-2) + (x.transpose(1,2) * causal_mask.unsqueeze(-1)).sum(dim=-2) # B x Nmax x C
    node_offdiag = (node_offdiag / 2)  / (num_elements+1e-10) # B x Nmax x C

    node = torch.cat([node_diag, node_offdiag], dim=-1) * node_mask.unsqueeze(-1)  # B x Nmax x 2C
    return node 

def blockwise_node_aggregation(node, block_id, virtual_mask, cumsum=True):
    """ Aggregate nodes withn the same block_id to blocks. Used for blockwise property prediction 
    node:         B x Nmax x C
    block_id:     B x Nmax
    virtual_mask: B x Nmax

    return: B x num_real_block x C
    """
    node_mask = get_node_mask_from_block_id(block_id)   # B x Nmax
    # aggregate from nodes to real_blocks and virtual_blocks
    real_block_mask = node_mask & ~virtual_mask         # B x Nmax
    num_real_blocks = block_id.max(dim=-1)[0] + 1       # B 
    offset = torch.cat([torch.zeros_like(num_real_blocks[:1]), num_real_blocks.cumsum(dim=0)[:-1]], dim=0) # B
    new_block_id = block_id + offset.unsqueeze(-1) # B x Nmax 

    block_mask = torch.arange(num_real_blocks.max(), device=node.device).unsqueeze(0) < num_real_blocks.unsqueeze(-1) # B x max_num_real_block
    
    real_block_agg = torch.zeros(node.shape[0], num_real_blocks.max(), node.shape[-1], device=node.device)
    real_block_agg[block_mask] = scatter_add(node[real_block_mask], new_block_id[real_block_mask], dim=0)             # B x max_num_real_block x C

    # add nodes from eariler blocks
    out_real = real_block_agg * block_mask.unsqueeze(-1) # B x max_num_real_block x C
    cumsum_real = real_block_agg.cumsum(dim=1) * block_mask.unsqueeze(-1) # B x max_num_real_block x C
    out_virtual = cumsum_virtual = None 
    if virtual_mask.any(): 
        """
        when use for inference, this is looking like use no virtual block! The only virtual block can be viewed as the last real block.
        """
        virtual_block_agg = torch.zeros(node.shape[0], num_real_blocks.max(), node.shape[-1], device=node.device) # assume that num_virtual_block = num_real_block
        virtual_block_agg[block_mask] = scatter_add(node[virtual_mask], new_block_id[virtual_mask], dim=0) # B x max_num_real_block x C
        out_virtual = virtual_block_agg * block_mask.unsqueeze(-1) # B x max_num_real_block x C
        cumsum_virtual = virtual_block_agg + torch.cat([torch.zeros_like(cumsum_real[:,0]).unsqueeze(1), cumsum_real[:,1:]], dim=1) 
        cumsum_virtual = cumsum_virtual * block_mask.unsqueeze(-1) # B x max_num_real_block x C

    return cumsum_real if cumsum else out_real, cumsum_virtual if cumsum else out_virtual # B x max_num_real_block x C

def get_real_virtual_edge_mask(causal_mask, virtual_mask):
    # causal_mask: B x Nmax x Nmax
    # virtual_mask: B x Nmax
    # we only return lower triangular part, hence mask out the upper triangular part of causal mask
    N_max = causal_mask.shape[-1]
    idx1, idx2 = torch.triu_indices(N_max, N_max, offset=0) # include diagonal part 
    tmp_mask = causal_mask.clone()
    tmp_mask[..., idx1, idx2] = False
    # split causal_mask into real and virtual part
    real_edge_mask    = tmp_mask & ~virtual_mask.unsqueeze(-1)
    virtual_edge_mask = tmp_mask & virtual_mask.unsqueeze(-1)
    return real_edge_mask, virtual_edge_mask # B x Nmax x Nmax


def blockwise_stats(block_size):
    # block_size: B x max_num_blocks
    block_mask = block_size > 0 # B x max_num_blocks
    num_blocks = block_mask.sum(dim=-1) # B
    total_num_blocks, max_num_blocks = num_blocks.sum(), num_blocks.max()
    block_node_size = block_size[:, :max_num_blocks] # B x max_num_blocks
    block_mask = block_mask[:, :max_num_blocks]      # B x max_num_blocks
    previous_block_node_size = torch.cat([torch.zeros_like(block_node_size[:, :1]), block_node_size[:, :-1]], dim=-1) # B x max_num_blocks
    previous_cumsum_block_node_size = previous_block_node_size.cumsum(dim=-1) # B x max_num_blocks

    # the computation below assumes the causal mask is generated based on the block_id. Also assumes:
    # 1. the the diagonal edge is not considered 
    # 2. only consider the lower-diagonal part 
    block_edge_size_diag = block_node_size * (block_node_size - 1) // 2   # B x max_num_blocks, doesn't consider self loop 
    block_edge_size_offdiag = block_node_size * previous_cumsum_block_node_size # B x max_num_blocks
    block_edge_size = block_edge_size_diag + block_edge_size_offdiag      # B x max_num_blocks
    block_edge_size = block_edge_size * block_mask                        # B x max_num_blocks

    flatten_num_nodes_withinblock = block_node_size[block_mask] # total_num_blocks x 1
    flatten_num_edges_withinblock = block_edge_size[block_mask] # total_num_blocks x 1
    return flatten_num_nodes_withinblock, flatten_num_edges_withinblock

def spread_blocktime_to_nodes_edges(block_time, block_id, block_size, virtual_mask):
    """
    block_time: total_num_blocks_in_batch x 1, only consider real or virtual blocks, not both. 
    block_id:     B x Nmax
    virtual_mask: B x Nmax 
    block_size:   B x max_num_blocks (num of real_blocks, not both)

    return = nodes_time, edges_time. Only for virtual blocks, real_blocks are with time 0, correspondingt to x0. 
    """
    # from flatten block_time to batch block_time 
    batch_block_time = torch.zeros_like(block_size) # B x max_num_real_blocks 
    batch_block_time[block_size > 0] = block_time   # B x max_num_real_blocks 

    # from batch block_time to nodes_time 
    nodes_time = spread_blockproperty_to_nodes(batch_block_time, block_id) # B x Nmax 
    # mask out the real nodes 
    nodes_time[~virtual_mask] = 0 # B x Nmax

    # from batch block_time to edges_time
    diag_causal_mask, offdiag_causal_mask = get_causal_block_mask(block_id, virtual_mask, separate_diag_offdiag=True) # B x Nmax x Nmax
    edges_time_diag = nodes_time.unsqueeze(-1) * diag_causal_mask # B x Nmax x Nmax
    edges_time_offdiag = nodes_time.unsqueeze(-1) * offdiag_causal_mask # B x Nmax x Nmax
    edges_time = edges_time_diag + edges_time_offdiag + edges_time_offdiag.transpose(1,2) # B x Nmax x Nmax
    return nodes_time, edges_time # B x Nmax, B x Nmax x Nmax 

def prepare_targetblock_transforms(target_block_mask, block_size=None):
    """
    target_block_mask: B x Nmax (edge_mask=False) or B x Nmax x Nmax (edge_mask=True). 
    target_block_mask can be either upper triangular matrix or lower traigular matrix.

    When block_size is provided, we batch in a different way, batch all blocks together instead of all graphs. 
    B will = sum of number of blocks in B graphs. 
    nmax will = max block (node/edge) size in B graphs. Can be done. 
    """
    assert target_block_mask.dim() in [2, 3]
    edge_mask = target_block_mask.dim() == 3
    device = target_block_mask.device

    if block_size is None:      
        num_elements = target_block_mask.sum(dim=list(range(target_block_mask.dim()))[1:])  # B
    else:
        num_elements = blockwise_stats(block_size)[target_block_mask.dim()-2]               # total_num_blocks

    batch_size = num_elements.size(0) # B or total_num_blocks
    max_elements = num_elements.max()
    assert target_block_mask.sum() == num_elements.sum()

    targetblock_padding_mask = torch.arange(max_elements, device=device) >= num_elements.unsqueeze(-1) # (B x m_max) or (total_mum_blocks x m_max)
    row_batch_targetblock, col_batch_targetblock = (~targetblock_padding_mask).nonzero(as_tuple=True)
    def from_full_to_batch_targetblock(full, flatten_input=False):
        # full: B x Nmax x Nmax x C or B x Nmax x C
        # flatten_input: sum x C 
        # out : B x m_max x C 
        batch_targetblock = torch.zeros(batch_size, max_elements, full.size(-1), device=device, dtype=full.dtype)
        batch_targetblock[row_batch_targetblock, col_batch_targetblock] = full[target_block_mask] if not flatten_input else full
        return batch_targetblock

    def from_batch_targetblock_to_full(full, batch_targetblock):
        flatten = batch_targetblock[row_batch_targetblock, col_batch_targetblock]
        full[target_block_mask] = flatten
        if edge_mask: # symmetric edge input 
            full.transpose(1, 2)[target_block_mask] = flatten # Must do transpose first. 
            assert (full == full.transpose(1, 2)).all()
        return full
    return from_full_to_batch_targetblock, from_batch_targetblock_to_full, targetblock_padding_mask




import os
import re

def find_checkpoint_with_lowest_val_loss(directory):
    lowest_val_loss = float('inf')
    best_checkpoint = None
    # Regex to match the pattern in the filename and extract the loss value
    pattern = re.compile(r'epoch=\d+-val_loss=([\d.]+)\.ckpt')

    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            val_loss = float(match.group(1))
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_checkpoint = filename
    if best_checkpoint is not None:
        return os.path.join(directory, best_checkpoint)
    else:
        return "No matching checkpoint found."
