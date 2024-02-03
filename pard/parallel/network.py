import torch.nn as nn
import torch, numpy as np
import torch.nn.functional as F 
from .layer import MaskedLayerNorm, MaskedBatchNorm

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class MLP(nn.Module):
    def __init__(self, idim:int, odim:int, hdim:int=None, channel_first:bool=False, nd=2, norm='none'):
        super().__init__()
        hdim = hdim or odim
        self.linear1 = conv_nd(nd, idim, hdim, kernel_size=1, padding=0, bias=True) if channel_first else nn.Linear(idim, hdim)
        self.linear2 = conv_nd(nd, hdim, odim, kernel_size=1, padding=0, bias=True) if channel_first else nn.Linear(hdim, odim)
        self.norm = MaskedBatchNorm(hdim, channel_first=channel_first) if norm == 'bn' else \
                    MaskedLayerNorm(hdim, channel_first=channel_first) if norm == 'ln' else lambda x,y:x

    def forward(self, x, mask=None):
        x = self.linear1(x)
        x = self.norm(x, mask) 
        x = F.silu(x) 
        x = self.linear2(x)
        return x
    
class PPGNBlock(nn.Module):
    def __init__(self, channel, norm='bn', add_transpose=False, simplify=False, prenorm=False):
        super().__init__()
        self.add_transpose = add_transpose
        self.prenorm = prenorm
        factor = 2 if add_transpose else 1
        self.norm1 = MaskedBatchNorm(channel, channel_first=True) if norm == 'bn' else \
                     MaskedLayerNorm(channel, channel_first=True) if norm == 'ln' else lambda x,y:x
        self.norm2 = MaskedBatchNorm(channel, channel_first=True) if norm == 'bn' else \
                     MaskedLayerNorm(channel, channel_first=True) if norm == 'ln' else lambda x,y:x
        if simplify:
            self.mlp1 = nn.Conv2d(factor*channel, channel, kernel_size=1, padding=0, bias=True)
            self.mlp2 = nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True)
        else:
            self.mlp1 = MLP(factor*channel, channel, hdim=channel, channel_first=True)
            self.mlp2 = MLP(channel, channel, hdim=channel, channel_first=True)

        self.linear = nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True)
        self.mlp3 = MLP(channel, channel, hdim=2*channel, channel_first=True)

    def forward(self, x, mask, causal_mask=None):
        """ PreNorm structure, PPGN + FFM 
            x    : B x D x N x N 
            mask : B x N x N 
            casual_block: None or  B x N x N, 
        """
        # PPGN
        xin = self.norm1(x, mask)          
        x1 = xin if not self.add_transpose else torch.cat([xin, xin.transpose(-1,-2)], dim=1)
        x1 = F.silu(self.mlp1(x1))  * mask.unsqueeze(1) 
        x2 = F.silu(self.mlp2(xin)) * mask.unsqueeze(1)
        if causal_mask is None:
            x_mult = torch.matmul(x1, x2) 
        else:
            x1_causal = x1 * causal_mask.unsqueeze(1)
            x2_causal = x2 * (causal_mask.unsqueeze(1).transpose(-1,-2))
            x_mult = torch.matmul(x1_causal, x2) + torch.matmul(x1, x2_causal) - torch.matmul(x1_causal, x2_causal) 
        x = self.linear(x_mult) * mask.unsqueeze(1) +  (x if self.prenorm else xin)

        # FFN
        xin = self.norm2(x, mask)          
        x = self.mlp3(xin, mask) * mask.unsqueeze(1) + (x if self.prenorm else xin)
        return x

from einops import rearrange
class TransformerBlock(nn.Module):
    def __init__(self, node_channel, edge_channel, n_head, norm='bn', prenorm=False):
        super().__init__()
        self.node_norm1 = MaskedBatchNorm(node_channel, channel_first=True) if norm == 'bn' else \
                     MaskedLayerNorm(node_channel, channel_first=True) if norm == 'ln' else lambda x,y:x
        self.edge_norm = MaskedBatchNorm(edge_channel, channel_first=True) if norm == 'bn' else \
                        MaskedLayerNorm(edge_channel, channel_first=True) if norm == 'ln' else lambda x,y:x
        self.node_norm2 = MaskedBatchNorm(node_channel, channel_first=True) if norm == 'bn' else \
                     MaskedLayerNorm(node_channel, channel_first=True) if norm == 'ln' else lambda x,y:x
        self.qk_node = conv_nd(1, node_channel, 2*edge_channel, kernel_size=1, padding=0)
        self.qk_edge = conv_nd(2, edge_channel, 2*edge_channel, kernel_size=1, padding=0)
        self.v_node = conv_nd(1, node_channel, node_channel, kernel_size=1, padding=0)
        self.v_edge = conv_nd(2, edge_channel, node_channel, kernel_size=1, padding=0)
        self.o_node = conv_nd(1, node_channel, node_channel, kernel_size=1, padding=0)
        self.o_edge = conv_nd(2, edge_channel, edge_channel, kernel_size=1, padding=0)
        self.n_head = n_head
        self.prenorm = prenorm
        self.att_linear = conv_nd(2, edge_channel, n_head, kernel_size=1, padding=0) 
        self.mlp = MLP(node_channel, node_channel, hdim=2*node_channel, channel_first=True, nd=1)

    def forward(self, X, E, edge_mask, edge_causal_mask=None):
        ## Modified based on "Graph Inductive Biases in Transformers without Message Passing"
        # https://arxiv.org/pdf/2305.17589.pdf
        # X: B x Dn x N 
        # E: B x De x N x N
        # edge_mask:        B x N x N, symmetric
        # edge_causal_mask: B x N x N, asymmetric, used in attention matrix 
        # :return  X, E 

        # create node_mask
        node_mask = edge_mask.any(dim=-1) # B x N
        
        X_norm = self.node_norm1(X, node_mask)
        X_qk = self.qk_node(X_norm)
        X_q, X_k= torch.split(X_qk, X_qk.shape[1]//2, dim=1) # B x De x N
        E_norm = self.edge_norm(E, edge_mask)
        E_qk = self.qk_edge(E_norm)
        E_q, E_k = torch.split(E_qk, E_qk.shape[1]//2, dim=1) # B x De x N x N

        E_new = X_q.unsqueeze(-1) + X_k.unsqueeze(-2) # B x De x N x N
        E_new = F.silu(E_k + F.silu(E_new)*E_q + E_new)*edge_mask.unsqueeze(1)
        E = (E if self.prenorm else E_norm) + self.o_edge(E_new)*edge_mask.unsqueeze(1) 

        # compute attention matrix, B x n_head x N x N 
        att = self.att_linear(E_new) # B x n_head x N x N

        assert att.isnan().sum() == 0, "att has nan"
        if edge_causal_mask is None:
            edge_causal_mask = edge_mask # this only mask the padding nodes and edges
        # need to use causal mask for attention matrix 
        # att = att + (~edge_causal_mask.unsqueeze(1) * -np.inf)
        att = att.masked_fill(~edge_causal_mask.unsqueeze(1), -np.inf)
        att = F.softmax(att, dim=-1) # B x n_head x N x N
        att = att.nan_to_num()
        assert att[:,0][~edge_causal_mask].sum() == 0, att[:,0][~edge_causal_mask].sum()

        X_v = self.v_node(X_norm) * node_mask.unsqueeze(1)   # B x Dn x N
        E_v = self.v_edge(E_new)  * edge_mask.unsqueeze(1)   # B x Dn x N x N
        X_v = rearrange(X_v, "b (h d) n -> b h n d", h=self.n_head)         # B x n_head x N x Dn//n_head)    
        E_v = rearrange(E_v, "b (h d) n1 n2 -> b h n1 n2 d", h=self.n_head) # B x n_head x N x N x Dn//n_head)      

        # Att@X + row_sum(Att elementwise_dot E), this is similar to GINE's formulation 
        X_new = (att @ X_v) + (att.unsqueeze(-1) * E_v).sum(dim=-2) # B x n_head x N x Dn//n_head
        X_new = rearrange(X_new, "b h n d -> b (h d) n")            # B x Dn x N
        X_new = self.o_node(X_new) * node_mask.unsqueeze(1)         # B x Dn x N
        X = (X if self.prenorm else X_norm)  + X_new 

        # FFN
        X_norm = self.node_norm2(X, node_mask)
        X = (X if self.prenorm else X_norm) + self.mlp(X_norm) * node_mask.unsqueeze(1)
        return X, E
    
class PPGNTransformerBlock(nn.Module):
    def __init__(self, node_channel, edge_channel, n_head, norm='bn', add_transpose=False, prenorm=True):
        super().__init__()
        self.ppgn = PPGNBlock(edge_channel, norm=norm, add_transpose=add_transpose, prenorm=prenorm, simplify=False)
        self.transformer = TransformerBlock(node_channel, edge_channel, n_head, norm=norm, prenorm=prenorm)

    def forward(self, X, E, edge_mask, edge_causal_mask):
        X, E = self.transformer(X, E, edge_mask, edge_causal_mask)
        E = self.ppgn(E, edge_mask, edge_causal_mask)
        return X, E

class PPGN(nn.Module):
    def __init__(self, channel, num_layers, norm='bn', add_transpose=False, prenorm=True, simplify=False):
        super().__init__()
        self.blocks = nn.ModuleList([PPGNBlock(channel, norm=norm, add_transpose=add_transpose, simplify=simplify, prenorm=prenorm) for _ in range(num_layers)])
        self.norm = MaskedBatchNorm(channel, channel_first=False) if norm == 'bn' else MaskedLayerNorm(channel, channel_first=False)
        self.out_mlp = MLP(channel, channel, channel_first=False)

    def forward(self, x, mask, causal_mask=None):
        # change x from B x N x N x D to B x D x N x N
        x = x.permute(0, 3, 1, 2)
        if causal_mask is not None:
            mask = mask & (causal_mask | causal_mask.transpose(-1, -2))  # B x N x N
        x = x * mask.unsqueeze(1) # B x D x N x N
        for block in self.blocks:
            x = block(x, mask, causal_mask)
        x = x.permute(0, 2, 3, 1) 
        x = self.out_mlp(self.norm(x, mask)) * mask.unsqueeze(-1)
        return x # B x N x N x D
    
class PPGNTransformer(nn.Module):
    def __init__(self, node_channel, edge_channel, num_layers, n_head, norm='bn', add_transpose=False, prenorm=True):
        super().__init__()
        self.blocks = nn.ModuleList([PPGNTransformerBlock(node_channel, edge_channel, n_head, norm=norm, add_transpose=add_transpose, prenorm=prenorm) for _ in range(num_layers)])
        self.node_norm = MaskedBatchNorm(node_channel, channel_first=False) if norm == 'bn' else MaskedLayerNorm(node_channel, channel_first=False)
        self.edge_norm = MaskedBatchNorm(edge_channel, channel_first=False) if norm == 'bn' else MaskedLayerNorm(edge_channel, channel_first=False)
        self.node_mlp = MLP(node_channel, node_channel, channel_first=False)
        self.edge_mlp = MLP(edge_channel, edge_channel, channel_first=False)

    def forward(self, X, E, edge_mask, edge_causal_mask):
        # X: B x N x D
        # E: B x N x N x D
        X, E = X.transpose(1,2), E.permute(0, 3, 1, 2)
        if edge_causal_mask is not None:
            edge_mask = edge_mask & (edge_causal_mask | edge_causal_mask.transpose(-1, -2))  # B x N x N
        node_mask = edge_mask.any(dim=-1) # B x N
        X, E = X * node_mask.unsqueeze(1), E * edge_mask.unsqueeze(1)
        for block in self.blocks:
            X, E = block(X, E, edge_mask, edge_causal_mask)
        X, E = X.transpose(1,2), E.permute(0, 2, 3, 1)
        X = self.node_mlp(self.node_norm(X, node_mask)) * node_mask.unsqueeze(-1)
        E = self.edge_mlp(self.edge_norm(E, edge_mask)) * edge_mask.unsqueeze(-1)
        return X, E
