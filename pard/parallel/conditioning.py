import torch
import torch.nn as nn 
from .utils import get_node_mask_from_block_id, spread_blockproperty_to_nodes, get_causal_block_mask, spread_blocktime_to_nodes_edges, get_edge_mask_from_block_id
from abc import abstractmethod


"""
    Input Conditioning Layers 
"""

class ConditioningBlock(nn.Module):
    @abstractmethod
    def forward(self, x, batch):
        ...
        
class IdentityConditioning(ConditioningBlock):
    def forward(self, x, batch):
        return x

class ConditioningSequential(nn.Sequential, ConditioningBlock):
    def forward(self, x, batch):
        for module in self:
            if isinstance(module, ConditioningBlock):
                x = module(x, batch)
            else:
                x = module(x)
        return x


from .structure_feature import StructureFeatures
class StructureFeatureConditioning(ConditioningBlock):
    def __init__(self, node_channels, k_eigval=5, k_eigvec=4, type='cycles'): # only cycles
        super().__init__()
        self.structure_feature = StructureFeatures(extra_features_type=type, k_eigval=k_eigval, k_eigvec=k_eigvec)
        dim = self.structure_feature.calculate_dims()[0]
        self.layer1 = nn.Linear(dim, node_channels)
        self.layer2 = nn.Linear(node_channels, node_channels)

    def forward(self, x, batch):
        X, E, block_id = batch.nodes, batch.edges, batch.nodes_blockid
        mask = get_node_mask_from_block_id(block_id)
        sf = self.structure_feature(X, E, mask)[0]
        sf = F.silu(self.layer1(sf))
        sf = self.layer2(sf)
        return x + sf * mask.unsqueeze(-1)
    
class StructureFeatureEdgeConditioning(ConditioningBlock):
    def __init__(self, edge_channels, k_eigval=5, k_eigvec=4, type='all'):
        super().__init__()
        self.structure_feature = StructureFeatures(extra_features_type=type, k_eigval=k_eigval, k_eigvec=k_eigvec)
        dim = self.structure_feature.calculate_dims()[0]
        self.layer1 = nn.Linear(dim, edge_channels)
        self.layer2 = nn.Linear(edge_channels, edge_channels)
    def forward(self, edge, batch):
        X, E, block_id = batch.nodes, batch.edges, batch.nodes_blockid
        mask = get_node_mask_from_block_id(block_id) # B x Nmax
        sf = self.structure_feature(X, E, mask)[0]   # B x Nmax x D
        sf_edge = (sf.unsqueeze(1) - sf.unsqueeze(2)).abs() # B x Nmax x Nmax x D
        sf_edge = F.silu(self.layer1(sf_edge)) # B x Nmax x Nmax x edge_channels
        sf_edge = self.layer2(sf_edge) # B x Nmax x Nmax x edge_channels
        # apply mask at the end
        edge_mask = get_edge_mask_from_block_id(block_id, batch.virtual_node_mask) # B x Nmax x Nmax
        return edge + sf_edge * edge_mask.unsqueeze(-1) # B x Nmax x Nmax x edge_channels

class BlockIDConditioning(ConditioningBlock): # TODO: think about normalization based relative block ID, kind of divide by max_num_blocks? (make it continuous)
    def __init__(self, channels, max_num_blocks=30):
        super().__init__()
        self.block_id_embedding = nn.Embedding(max_num_blocks+1, channels) # block=-1 is also included 
    def forward(self, x, batch):
        block_id = batch.nodes_blockid
        mask = get_node_mask_from_block_id(block_id)
        return (x + self.block_id_embedding(block_id + 1)) * mask.unsqueeze(-1)
    
# Solution: edge based relative block ID encoding. 1/(|block_id1 - block_id2| + 1) @ normalize to 0-1
class RelativeBlockIDEdgeConditioning(ConditioningBlock):
    def __init__(self, hidden, edge_channels):
        super().__init__()
        self.relative_id_embedding = PositionalEmbedding(hidden, max_positions=10)
        self.layer1 = nn.Linear(hidden, 2*edge_channels)
        self.layer2 = nn.Linear(2*edge_channels, edge_channels)

    def forward(self, edge, batch):
        node_id = batch.nodes_blockid # B x Nmax
        relative_edge_id = torch.abs(node_id.unsqueeze(1) - node_id.unsqueeze(2)) # B x Nmax x Nmax
        relative_edge_id = 1 / (relative_edge_id + 1) # B x Nmax x Nmax

        # transform edge_id from float to embedding 
        relative_edge_encoding = self.relative_id_embedding(relative_edge_id) # B x Nmax x Nmax x hidden
        relative_edge_encoding = F.silu(self.layer1(relative_edge_encoding)) # B x Nmax x Nmax x edge_channels
        relative_edge_encoding = self.layer2(relative_edge_encoding) # B x Nmax x Nmax x edge_channels

        # apply mask at the end
        edge_mask = get_edge_mask_from_block_id(node_id, batch.virtual_node_mask) # B x Nmax x Nmax
        return edge + relative_edge_encoding * edge_mask.unsqueeze(-1) # B x Nmax x Nmax x edge_channels
        

class BlockDegreeConditioning(ConditioningBlock):
    def __init__(self, channels, max_degree=10):
        super().__init__()
        self.block_degree_embedding = nn.Embedding(max_degree, channels) 

    def forward(self, x, batch):
        block_degree = batch.block_degree      # B x max_num_blocks
        block_id = batch.nodes_blockid         # B x max_num_nodes 
        node_block_degree = spread_blockproperty_to_nodes(block_degree, block_id) # B x Nmax
        mask = get_node_mask_from_block_id(block_id)
        return (x + self.block_degree_embedding(node_block_degree)) * mask.unsqueeze(-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        # x: B x 1, 1d time tensor  
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device) # T 
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        freqs = freqs.to(x.dtype)
        # if x.dim() == 1:
        #     x = x.outer(freqs) # x = 1:1000
        # elif x.dim() ==2:
        #     x = torch.einsum('bi,t->bit', x, freqs) # x = B x 1:1000
        x = x.unsqueeze(-1) * freqs # 
        x = torch.cat([x.cos(), x.sin()], dim=-1) # B x D or B x Nmax x D
        return x
    
import torch.nn.functional as F   
class TimestepConditioning(ConditioningBlock):
    def __init__(self, temb_channels, channels, blockwise_timestep=False, only_for_target=False):
        super().__init__()
        self.time_embed = PositionalEmbedding(temb_channels)
        self.layer1 = nn.Linear(temb_channels, channels)
        self.layer2 = nn.Linear(channels, channels)
        self.blockwise_timestep = blockwise_timestep
        self.only_for_target = only_for_target  

    def get_time_embedding(self, timestep):
        temb = self.time_embed(timestep) # B x temb_channels
        temb = temb.reshape(temb.shape[0], 2, -1).flip(1).reshape(*temb.shape) 
        temb = F.silu(self.layer1(temb))
        temb = F.silu(self.layer2(temb))
        return temb      

    def forward(self, x, batch):
        # x: B x Nmax x D
        #### TODO 1. When t has shape (total_num_real_or_virtual_blocks, 1), the output is wrong.
        timestep = batch.timestep         # B or total_num_real_or_virtual_blocks 
        block_id = batch.nodes_blockid    # B x max_num_nodes 
        if self.blockwise_timestep:
            if timestep.size(0) != x.size(0):
                # training, total_num_real_blocks x 1
                timestep,_ = spread_blocktime_to_nodes_edges(timestep, 
                                                            batch.nodes_blockid, 
                                                            batch.block_size,
                                                            batch.virtual_node_mask) # B x Nmax
            else:
                # inference case in generation, set the timestep at real part to be 0 
                # get last block as virtual block 
                assert timestep.dim() == 1, timestep.shape
                timestep = timestep.unsqueeze(-1).expand(-1, x.size(1)) # B x Nmax
                last_mask = (block_id == block_id.max(dim=-1, keepdim=True)[0])      # B x N_max
                timestep[~last_mask] = 0

        mask = get_node_mask_from_block_id(block_id)
        temb = self.time_embed(timestep) # B x temb_channels or B x Nmax x temb_channels
        target_shape = temb.shape
        temb = temb.reshape(target_shape[:-1]+ (2, -1)).flip(-2).reshape(*target_shape)  # B x Nmax x temb_channels 
        temb = F.silu(self.layer1(temb))
        temb = self.layer2(temb)
        if temb.dim() != x.dim():
            temb = temb.unsqueeze(1).expand(-1, x.size(1), -1) # B x Nmax x D
        if self.only_for_target:
            # only add the time embedding to nodes within the target block, which is the virtual node
            temb = temb * batch.virtual_node_mask.unsqueeze(-1)
            # temb[~batch.virtual_node_mask] = 0
        return (x + temb) * mask.unsqueeze(-1)

class NodeEdgeCombiner(nn.Module):
    # outputs are not masked 
    def __init__(self, node_channels, edge_channels, out_channels, simple=False):
        super().__init__()
        self.node_layer1 = nn.Linear(node_channels, 2*out_channels) 
        self.edge_layer1 = nn.Linear(edge_channels, out_channels) 
        if simple:
            self.fuse_layer = nn.Linear(2*out_channels, out_channels)
        else:
            self.fuse_layer1 = nn.Linear(out_channels, out_channels) 
            self.fuse_layer2 = nn.Linear(out_channels, out_channels)
        self.simple = simple

    def forward(self, nodes, edges):
        # nodes: B x Nmax x Dn
        # edges: B x Nmax x Nmax x De
        # compute node and edge features
        edge = self.edge_layer1(edges)
        node = self.node_layer1(nodes)
        node1, node2 = torch.split(node, node.shape[2]//2, dim=2)
        node = node1.unsqueeze(1) + node2.unsqueeze(2) # B x Nmax x Nmax x C
        if self.simple:
            edge = F.silu(self.fuse_layer(F.silu(torch.cat([node, edge], dim=-1)))) # B x Nmax x Nmax x C
        else:
            edge = F.silu(node + edge + F.silu(self.fuse_layer1(node)) * F.silu(self.fuse_layer2(edge)) )
        return edge # B x Nmax x Nmax x C
    
class NodeEdgeEncoder(nn.Module):
    # outputs are not masked 
    def __init__(self, num_node_features, num_edge_features, node_channels, edge_channels):
        super().__init__()
        self.node_encoder = nn.Linear(num_node_features, node_channels)
        self.edge_encoder = nn.Linear(num_edge_features, edge_channels)
    def forward(self, batch):
        return self.node_encoder(batch.nodes), self.edge_encoder(batch.edges)
    
"""
   Parallel Output Conditioning Layers 
"""
from .utils import edge_to_node, blockwise_node_aggregation
from .network import MLP

class BlockOutDecoder(nn.Module):
    def __init__(self, node_channels, edge_channels, max_block_size, max_block_degree, norm='bn', include_previous_block=True):
        super().__init__()
        self.transform = nn.Sequential(nn.Linear(2*edge_channels+node_channels, node_channels), nn.SiLU())
        self.block_size_out     = MLP(node_channels, max_block_size,   node_channels, norm=norm) 
        self.block_degree_out   = MLP(node_channels, max_block_degree, node_channels, norm=norm) 
        self.init_block_degree_out = MLP(node_channels, max_block_degree, node_channels, norm=norm) 
        self.block_size_encoder = nn.Embedding(max_block_size, node_channels)
        self.cumsum = include_previous_block

    def forward(self, node, edge, batch, return_only_first_block=False, blocksize_target=None):
        """
        node: B x Nmax x C
        edge: B x Nmax x Nmax x C 
        """
        block_size = batch.block_size        # B x max_num_blocks
        # encode the first block 
        first_block_size = block_size[:, 0] # B
        first_blockdegree_pred = self.init_block_degree_out(self.block_size_encoder(first_block_size)) # B x max_block_degree
        first_blockdegree_target = batch.block_degree[:, 0] if hasattr(batch, 'block_degree') else None
        if return_only_first_block:
            return first_blockdegree_pred, first_blockdegree_target

        # inputs 
        block_id     = batch.nodes_blockid     # B x Nmax
        virtual_mask = batch.virtual_node_mask # B x Nmax
        block_degree = batch.block_degree      # B x max_num_blocks

        # get block mask 
        block_mask = block_size > 0 # B x max_num_blocks
        node_mask = get_node_mask_from_block_id(block_id) # B x Nmax    

        # Aggregate to blockwise representation 
        x = edge_to_node(edge, block_id, virtual_mask)                   # B x Nmax x 2De
        x = torch.cat([x, node], dim=-1)                                 # B x Nmax x 2De+Dn
        x = self.transform(x)                                            # B x Nmax x Dn
        x = x * node_mask.unsqueeze(-1)                                  # B x Nmax x Dn

        block_representation, _ = blockwise_node_aggregation(x, block_id, virtual_mask, self.cumsum)  # B x max_num_blocks x C

        assert block_representation[~block_mask].sum() == 0, block_representation[~block_mask].sum()
        assert block_representation.shape[:-1] == block_size.shape 

        ## seems the below assert is not necessary
        # assert block_size.size(1) >= 2, "max_num_blocks should be at least 2 for shifting 1 position" 

        # prepare block_size 
        blocksize_pred = self.block_size_out(block_representation, block_mask)      # B x max_num_blocks x max_block_size
        # next one 
        if blocksize_target is None:
            blocksize_target = block_size.clone()
            blocksize_target[:, :-1] = blocksize_target[:, 1:]
            blocksize_target[:, -1] = 0

        # prepare block_degree, conditional on encoding of block size. THis doesn't predict the first block degree. 
        blockdegree_pred = self.block_degree_out(block_representation + self.block_size_encoder(blocksize_target), block_mask) # B x max_num_blocks x max_block_degree
        blockdegree_target = block_degree.clone()
        blockdegree_target[:, :-1] = blockdegree_target[:, 1:]
        blockdegree_target[:, -1] = 0

        return blocksize_pred, blocksize_target, blockdegree_pred, blockdegree_target, block_mask, first_blockdegree_pred, first_blockdegree_target 
    

class VirtualBlockOutDecoder(nn.Module):
    def __init__(self, node_channels, edge_channels, num_node_classes, num_edge_classes, norm='bn', Tmax=0):
        super().__init__()
        self.node_transform = nn.Sequential(nn.Linear(2*edge_channels+node_channels, node_channels), nn.SiLU())
        self.edge_transforom = NodeEdgeCombiner(node_channels, edge_channels, 2*edge_channels)
        self.node_out = MLP(node_channels, num_node_classes, node_channels, norm=norm)
        self.edge_out = MLP(2*edge_channels, num_edge_classes, edge_channels, norm=norm)
        self.Tmax = Tmax
        if Tmax > 0: # Tmax=0 to disable input conditioning
            self.phi_node = MonotonicMLP(128, 2)
            self.phi_edge = MonotonicMLP(128, 2)

    def forward(self, node, edge, batch, causal=True):
        """
        node: B x Nmax x F
        edge: B x Nmax x Nmax x C 
        """
        timestep     = batch.timestep           # B
        block_id     = batch.nodes_blockid      # B x Nmax
        virtual_mask = batch.virtual_node_mask  # B x Nmax
        node_input   = batch.nodes              # B x Nmax x F
        edge_input   = batch.edges              # B x Nmax x Nmax x C

        # get masks
        node_mask = get_node_mask_from_block_id(block_id) # B x Nmax
        causal_mask = get_causal_block_mask(block_id, virtual_mask) # B x Nmax x Nmax
        edge_mask = causal_mask | causal_mask.transpose(-1, -2)     # B x Nmax x Nmax
        node = node * node_mask.unsqueeze(-1)                       # B x Nmax x F
        edge = edge * edge_mask.unsqueeze(-1)                       # B x Nmax x Nmax x C

        # aggregate edges to node representation 
        x = edge_to_node(edge, block_id, virtual_mask, causal=causal)                      # B x Nmax x 2De
        node = torch.cat([x, node], dim=-1)                                 # B x Nmax x 2De+Dn
        node = self.node_transform(node)                                    # B x Nmax x Dn
        node = node * node_mask.unsqueeze(-1)                               # B x Nmax x Dn

        # update edge to new representation by combining the node representation 
        edge = self.edge_transforom(node, edge)                             # B x Nmax x Nmax x 2De
        edge = edge + edge.transpose(1,2)                                   # B x Nmax x Nmax x 2De
        edge = edge * edge_mask.unsqueeze(-1)                               # B x Nmax x Nmax x 2De

        node_logit = self.node_out(node, node_mask)                         # B x Nmax x F
        edge_logit = self.edge_out(edge, edge_mask)                         # B x Nmax x Nmax x C

        # input conditioning here: directly process logit instead of dist 
        if self.Tmax > 0:
            batch_size = node_logit.size(0)
            # here we support batchwise time or blockwise time
            if timestep.size(0) != batch_size: 
                # blockwise time: total_num_real_blocks x 1
                # TODO: the training and inference will be different, need to consider inference later. 
                ## seems that we don't need change for inference, it works with batchwise time at inference. 
                nodes_time, edges_time = spread_blocktime_to_nodes_edges(
                    timestep, 
                    batch.nodes_blockid, 
                    batch.block_size,
                    batch.virtual_node_mask
                ) # B x Nmax, B x Nmax x Nmax
                nodes_time = nodes_time.unsqueeze(-1) / self.Tmax # B x Nmax x 1
                edges_time = edges_time.unsqueeze(-1) / self.Tmax # B x Nmax x Nmax x 1
            else:
                # batchwise time: B
                nodes_time = (timestep / self.Tmax).view(batch_size, 1, 1) 
                edges_time = (timestep / self.Tmax).view(batch_size, 1, 1, 1)

            # node_logit += (1-nodes_time) * node_input
            # edge_logit += (1-edges_time) * edge_input
            node_logit = node_logit + self.phi_node(1-nodes_time) * node_input # phi should be monotonic, the larger the t, the smaller the value.
            edge_logit = edge_logit + self.phi_edge(1-edges_time) * edge_input # for stability, can normalize t to [0,1] first 

        return node_logit, edge_logit

class MonotonicLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_channels, out_channels))
        self.bias = nn.Parameter(torch.rand(out_channels))
    def forward(self, x):
        return x @ torch.abs(self.weight) + self.bias

class MonotonicMLP(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([MonotonicLayer(channels if i > 0 else 1, 1 if i == num_layers - 1 else channels) 
                                     for i in range(num_layers)])
    def forward(self, t):
        # t: B x 1
        for i, layer in enumerate(self.layers):
            t = layer(t)
            t = F.silu(t) if i != self.num_layers-1 else F.sigmoid(t)
        return t


"""
   Batched Sequential Output Conditioning Layers 
"""

class GraphOutDecoder(nn.Module):
    def __init__(self, node_channels, edge_channels, max_block_size, max_block_degree, norm='bn'):
        super().__init__()
        self.init_block_degree_out = MLP(node_channels, max_block_degree, node_channels, norm=norm) 
        self.transform = nn.Sequential(nn.Linear(2*edge_channels+node_channels, node_channels), nn.SiLU())
        self.block_size_out     = MLP(node_channels, max_block_size,   node_channels, norm=norm) 
        self.block_degree_out   = MLP(node_channels, max_block_degree, node_channels, norm=norm) 
        self.block_size_encoder = nn.Embedding(max_block_size, node_channels)

    def forward(self, node, edge, batch, return_only_first_block=False, blocksize_target=None):
        """
        node: B x Nmax x F
        edge: B x Nmax x Nmax x C 
        """
        block_size   = batch.block_size        # B x max_num_blocks
        # encode the first block 
        first_block_size = block_size[:, 0] # B
        first_blockdegree_pred = self.init_block_degree_out(self.block_size_encoder(first_block_size)) # B x max_block_degree
        # first_blockdegree_target = batch.block_degree[:, 0] if hasattr(batch, 'block_degree') else None
        if return_only_first_block:
            return first_blockdegree_pred # need to subsample to number of original graphs, or loss divide by number of partial graphs 
        
        block_id     = batch.nodes_blockid       # B x Nmax
        virtual_mask = batch.virtual_node_mask
        node_mask    = get_node_mask_from_block_id(block_id) # B x Nmax

        # Aggregate edge to node 
        x = edge_to_node(edge, block_id, virtual_mask, causal=False)     # B x Nmax x 2De
        x = torch.cat([x, node], dim=-1)                                 # B x Nmax x 2De+Dn
        x = self.transform(x)                                            # B x Nmax x Dn
        x = x * node_mask.unsqueeze(-1)                                  # B x Nmax x Dn

        # Aggregate to graph representation
        graph_representation = x.sum(dim=1)                              # B x Dn

        # prepare block_size 
        blocksize_pred = self.block_size_out(graph_representation)      # B x max_block_size
        if blocksize_target is None:
            blocksize_target = batch.next_block_size.squeeze(-1)        # B
        blockdegree_pred = self.block_degree_out(graph_representation + self.block_size_encoder(blocksize_target))  # B x max_block_degree
        return blocksize_pred, blockdegree_pred, first_blockdegree_pred