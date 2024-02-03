import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_position_embedding(length: int, embedding_dim: int):
    """Build sinusoidal embeddings.

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(length, dtype=torch.float).unsqueeze(
        1
    ) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb).unsqueeze(-1),
                     torch.cos(emb).unsqueeze(-1)],
                    dim=-1).view(length, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
    return emb

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000, time_scale=1000.0):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = time_scale * timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class MaskedLayerNorm(nn.Module):
    def __init__(self, dim:int, channel_first:bool=False):
        super().__init__()
        self.channel_first = channel_first
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        if self.channel_first:
            x = self.ln(x.transpose(1, -1)).transpose(1, -1)
            if mask is not None:
                if mask.dim() != x.dim():
                    mask = mask.unsqueeze(1)
                x = x * mask
        else:
            x = self.ln(x)
            if mask is not None:
                if mask.dim() != x.dim():
                    mask = mask.unsqueeze(-1)
                x = x * mask  
        # x = F.relu(x)      
        return x 

class MaskedBatchNorm(nn.Module):
    def __init__(self, dim:int, channel_first:bool=False, act=False):
        super().__init__()
        self.channel_first = channel_first
        self.bn = nn.BatchNorm1d(dim)
        self.act = act

    def forward(self, x, mask=None):
        assert x.dim() >= 2, "Input must be at least 2D"
        # assume channel is the last dimension
        if mask is not None:
            if self.channel_first:
                if x.dim() == 4:
                    mask = mask.squeeze(1) if mask.dim() == 4 else mask
                    midx = torch.nonzero(mask)
                    x_indexed = x[midx[:, 0], :, midx[:, 1], midx[:, 2]]
                    x[midx[:, 0], :, midx[:, 1], midx[:, 2]] = self.bn(x_indexed)
                elif x.dim() == 3:
                    # x : B x D x N 
                    mask = mask.squeeze(1) if mask.dim() == 3 else mask # B x N 
                    midx = torch.nonzero(mask)
                    x_indexed = x[midx[:, 0], :, midx[:, 1]]
                    x[midx[:, 0], :, midx[:, 1]] = self.bn(x_indexed)
                else:
                    raise NotImplementedError#, "Not implemented for dim = %d" % x.dim()
            else:
                x[mask] = self.bn(x[mask])
        elif x.dim() == 2:
            x = self.bn(x)
        else:
            x = self.bn(x) if self.channel_first else self.bn(x.flatten(end_dim=-2)).unflatten(dim=0, size=x.shape[:-1])
        if self.act:
            x = F.relu(x)
        return x

class MLP(nn.Module):
    def __init__(self, idim:int, odim:int, hdim:int=None, norm:str='none', channel_first:bool=False, one_layer:bool=False):
        super().__init__()
        hdim = hdim or idim
        self.one_layer = one_layer
        if one_layer:
            hdim = odim

        self.linear1 = nn.Conv2d(idim, hdim, kernel_size=1, padding=0, bias=True) if channel_first else nn.Linear(idim, hdim)
        if not one_layer:
            self.linear2 = nn.Conv2d(hdim, odim, kernel_size=1, padding=0, bias=True) if channel_first else nn.Linear(hdim, odim)
            assert norm in ['none', 'bn', 'ln']
            self.norm = None
            if norm == 'bn':
                self.norm = MaskedBatchNorm(hdim, channel_first=channel_first, act=False)
            elif norm == 'ln':
                self.norm = MaskedLayerNorm(hdim, channel_first=channel_first)

    def forward(self, x, mask=None):
        x = self.linear1(x)
        if not self.one_layer:
            if self.norm:
                x = self.norm(x, mask=mask) # use norm to replace activation function
            x = F.relu(x) 
            x = self.linear2(x)
        return x

from LocalDiffusion.utils import to_dense, to_dense_batch

class SparseTo2dDenseFormator:
    def __init__(self, extra_feature_func=None, one_hot=True) -> None:
        self.extra_feature_func = extra_feature_func or (lambda x: x)
        self.one_hot = one_hot

    def __call__(self, batch):
        batch = self.format(batch)
        return self.extra_feature_func(batch)
    
    def add_features(self, batch):
        return self.extra_feature_func(batch)
    
    def format(self, batch):        
        x, adj, mask = to_dense(batch, self.one_hot) # adj diag should be 0

        block = torch.zeros_like(batch.x[:,0], dtype=torch.long, device=x.device) if not hasattr(batch, "block") else batch.block
        block, _ = to_dense_batch(block, batch.batch, fill_value=-1.)
        block = block.long()    # [B, Nmax]

        del batch.x, batch.edge_index, batch.edge_attr, batch.batch,
        batch.nodes = x 
        batch.node_mask = mask
        batch.edges = adj
        batch.block = block
        # batch.graph is passed by default if batch has graph attribute， with dim [B, d_graph]
        return batch
    