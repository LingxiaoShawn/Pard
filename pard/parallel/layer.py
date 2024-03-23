import torch
import torch.nn as nn
import torch.nn.functional as F

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

    