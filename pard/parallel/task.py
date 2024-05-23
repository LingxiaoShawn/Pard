import lightning as L
import torch
import numpy as np 
from torch_geometric.data import Batch
from torch_scatter import scatter_max

from .utils import *
from .network import * 
from .conditioning import * 
from .optim import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from .transform import ToDenseParallelBlocksBatch
from pard.diffusion import sample_categorical, DiscreteDiffusion


class PredictBlockProperties(L.LightningModule):
    def __init__(self, 
            num_node_features, 
            num_edge_features, 
            max_num_blocks,
            max_block_size,
            max_block_degree,
            channels, num_layers, norm='bn', add_transpose=False, prenorm=True, # PPGN parameters
            edge_channels=0, n_head=0,         # PPGNTransformer's additional parameters 
            lr=0.001, wd=0.01, lr_patience=10, lr_warmup=5, lr_scheduler='plateau', lr_epochs=100, # optim parameters
            use_relative_blockid=False, # use relative ID can help reduce overfitting, as the absolute id is not meaningful
            use_absolute_blockid=True, # use absolute ID
            batched_sequential=False,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.to_dense_batch = ToDenseParallelBlocksBatch()
        self.input_encoder = NodeEdgeEncoder(num_node_features, num_edge_features, channels, edge_channels or channels)
        self.input_conditioner = ConditioningSequential(
            BlockIDConditioning(channels, max_num_blocks) if use_absolute_blockid else IdentityConditioning(),
            StructureFeatureConditioning(channels, k_eigval=5, k_eigvec=5) if batched_sequential else IdentityConditioning(),
        ) 
        self.edge_conditioner = ConditioningSequential(
            RelativeBlockIDEdgeConditioning(edge_channels or channels, edge_channels or channels),
        ) if use_relative_blockid else IdentityConditioning()

        if edge_channels > 0 and n_head > 0:
            self.use_transformer = True 
            self.net = PPGNTransformer(channels, edge_channels, num_layers, n_head, norm=norm, add_transpose=add_transpose, prenorm=prenorm)
        else:
            self.use_transformer = False  
            self.node_edge_combiner = NodeEdgeCombiner(channels, channels, channels)
            self.net = PPGN(channels, num_layers, norm=norm, add_transpose=add_transpose, simplify=False, prenorm=prenorm)

        # output conditioning 
        if batched_sequential:
            self.out_decoder = GraphOutDecoder(channels, edge_channels or channels, max_block_size, max_block_degree, 
                                            norm=norm)
        else:
            self.out_decoder = BlockOutDecoder(channels, edge_channels or channels, max_block_size, max_block_degree, 
                                            norm=norm, include_previous_block=True)

        self.lr = lr
        self.wd = wd
        self.lr_patience = lr_patience
        self.lr_warmup = lr_warmup  
        self.lr_scheduler = lr_scheduler
        self.lr_epochs = lr_epochs
        self.ignore_last = False
        self.batched_sequential = batched_sequential

    @torch.no_grad()
    def inference(self, batch, sparse=False, pred_init_degree=False, return_argmax=False):
        # Sequentially predict the next block's size and degree. Only the last block's prediction will be used.

        self.eval()
        if pred_init_degree:
            # predict the init block_size, input should contains the batched current block's prediction 
            assert (not sparse) and hasattr(batch, 'block_size') # dense format 
            if self.batched_sequential:
                pred = self.out_decoder(None, None, batch, return_only_first_block=True) # B x max_block_degree
            else:
                pred = self.out_decoder(None, None, batch, return_only_first_block=True)[0] # B x max_block_degree
            dist = F.softmax(pred, dim=-1) # distribution
            next_block_degree = dist.argmax(-1) if return_argmax else sample_categorical(dist) # B
            return next_block_degree  # B
        
        node, edge, batch = self.forward_representation(batch, sparse=sparse)
        batch_size = node.size(0)
        if self.batched_sequential:
            # predict next size 
            size_pred = self.out_decoder(node, edge, batch, return_only_first_block=False, blocksize_target=batch.block_size[:,0])[0]
            size_pred = F.softmax(size_pred, dim=-1) # B x Class
            next_block_size = size_pred.argmax(-1) if return_argmax else sample_categorical(size_pred)

            # predict next degree
            degree_pred = self.out_decoder(node, edge, batch, return_only_first_block=False, blocksize_target=next_block_size)[1]
            dist = F.softmax(degree_pred, dim=-1)
            next_block_degree = dist.argmax(-1) if return_argmax else sample_categorical(dist) # B
            return next_block_size, next_block_degree # B, B
        else:      
            # predict next size 
            out = self.out_decoder(node, edge, batch, return_only_first_block=False)
            size_pred, block_mask = out[0], out[4]
            size_pred = F.softmax(size_pred, dim=-1) # distribution, B x max_block_size x Class

            # get last block, by identify the first 0 in block_size
            block_size = batch.block_size                # B x max_num_blocks
            next_block_idx = block_mask.sum(dim=-1)      # B
            last_block_idx = next_block_idx - 1          # B
            idx1 = torch.arange(batch_size, device=next_block_idx.device)
            dist = size_pred[idx1, last_block_idx]      # B x Class
            next_block_size = dist.argmax(-1) if return_argmax else sample_categorical(dist) # B

            # predict next degree
            # 1. prepare target block_size 
            blocksize_target = block_size.clone()
            blocksize_target[:, :-1] = blocksize_target[:, 1:]
            blocksize_target[:, -1] = 0
            blocksize_target[idx1, last_block_idx] = next_block_size
            ### next_block_idx can become out of boundary when block_mask if full 
            ### Solution 1: provide the target block size as input to self.out_decoder.
            out = self.out_decoder(node, edge, batch, return_only_first_block=False, blocksize_target=blocksize_target)
            degree_pred, block_mask = out[2], out[4]
            dist = F.softmax(degree_pred, dim=-1)[idx1, last_block_idx] # B x max_block_degree
            next_block_degree = dist.argmax(-1) if return_argmax else sample_categorical(dist) # B
        return next_block_size, next_block_degree # B, B 

    def forward_representation(self, batch, sparse=True):
        # to dense 
        if sparse:
            batch = self.to_dense_batch(batch)
        edge_causal_mask = get_causal_block_mask(batch.nodes_blockid, batch.virtual_node_mask) if not self.batched_sequential else None 
        edge_mask = get_edge_mask_from_block_id(batch.nodes_blockid, batch.virtual_node_mask)
        # encode node and edge: batch.nodes and batch.edges can be used for output conditioning in diffuison 
        node, edge = self.input_encoder(batch)
        # node edge input conditioning
        node = self.input_conditioner(node, batch)
        edge = self.edge_conditioner(edge, batch)  # add edge conditioning
        if self.use_transformer:
            node, edge = self.net(node, edge, edge_mask, edge_causal_mask)
        else:
            edge = self.node_edge_combiner(node, edge)
            edge = self.net(edge, edge_mask, edge_causal_mask)
        return node, edge, batch

    def forward(self, batch, sparse=True, return_only_first_block=False):
        node, edge, batch = self.forward_representation(batch, sparse=sparse)
        if self.batched_sequential:
            if return_only_first_block:
                first_degree_pred = self.out_decoder(node, edge, batch, return_only_first_block)  
                first_degree_target = batch.block_degree[:, 0]
                return first_degree_pred, first_degree_target
            else:
                size_pred, degree_pred, first_degree_pred = self.out_decoder(node, edge, batch, return_only_first_block)
                size_target = batch.next_block_size.squeeze(-1)
                degree_target = batch.next_block_degree.squeeze(-1)
                first_degree_target = batch.block_degree[:, 0]
                mask = torch.ones_like(size_target, dtype=torch.bool)
                if hasattr(batch, 'to_original_graphs'):
                    _, argmax = scatter_max(first_degree_target, batch.to_original_graphs, dim=0)
                    first_degree_pred = first_degree_pred[argmax]
                    first_degree_target = first_degree_target[argmax]

                return size_pred, size_target, degree_pred, degree_target, mask, first_degree_pred, first_degree_target
        if return_only_first_block:
            first_degree_pred, first_degree_target = self.out_decoder(node, edge, batch, return_only_first_block)
            return first_degree_pred, first_degree_target
        else:
            size_pred, size_target, degree_pred, degree_target, mask, first_degree_pred, first_degree_target = self.out_decoder(node, edge, batch, return_only_first_block)
            return size_pred, size_target, degree_pred, degree_target, mask, first_degree_pred, first_degree_target
    
    def _shared_step(self, batch, split='train', ignore_last=False):
        size_pred, size_target, degree_pred, degree_target, mask, first_degree_pred, first_degree_target = self(batch)
        batch_size = first_degree_pred.size(0)  #### use number of original graph as batch size 
        if ignore_last:
            # This is used where the inputs are partial blocks, for testing the causal stability 
            # This should be combine with validation loader with the transform having random_partial_blocks=True
            last_mask_idx = mask.sum(dim=-1) - 1
            idx1 = torch.arange(batch_size, device=last_mask_idx.device)
            mask[idx1, last_mask_idx] = False

        size_loss = F.cross_entropy(size_pred[mask], size_target[mask], reduction='sum')
        degree_loss = F.cross_entropy(degree_pred[mask], degree_target[mask], reduction='sum')
        first_degree_loss = F.cross_entropy(first_degree_pred, first_degree_target, reduction='sum')
        loss = size_loss + degree_loss + first_degree_loss
        loss = loss / batch_size

        self.log(f"{split}_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{split}_size_acc", (size_pred[mask].argmax(dim=-1) == size_target[mask]).float().mean(), 
                                        prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{split}_degree_acc", (degree_pred[mask].argmax(dim=-1) == degree_target[mask]).float().mean(), 
                                        prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log(f"{split}_first_degree_acc", (first_degree_pred.argmax(dim=-1) == first_degree_target).float().mean(),
                                        prog_bar=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, split='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, split='val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, split='test', ignore_last=self.ignore_last)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.lr_warmup,
                                                        num_training_steps=self.lr_epochs,
                                                        min_lr=1e-6,
                                                        min_lr_mode="clamp")
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=self.lr_patience,
                                                                    verbose=True)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

class AutoregressiveDiffusion(L.LightningModule):
    """
    1. Input: dense or sparse batch. 
        - Virtual block nodes and edges are set with virtual type first. Then all these nodes and edges will be replaced 
          with sampled outcome from diffusion model. (equivalent to virtual_edge_mask and virtual_node_mask)
        - Conditioning: 
            * BlockID
            * BlockDegree 
            * Time 
    2. Diffusion model: think about how to support efficient same marginal distribution for all elements.
        - Later: time can be different across blocks. For now that's fine. 
    3. Output: 
        - _prepare_targetblock_transforms: change format between dense batch and sparse flatten batch.  
        - Conditioning:
            Think about add input with time as coefficient 
    4. Inference:
        - Support sequential single step generation 
    """
    def __init__(self,
            #------------- params for models -----------------
            num_node_features, 
            num_edge_features, 
            max_num_blocks,
            max_block_size,
            max_block_degree,
            channels, 
            num_layers, 
            norm='bn',
            prenorm=True,  
            add_transpose=False,               # PPGN parameters
            edge_channels=0, n_head=0,         # PPGNTransformer's additional parameters 
            use_input=False,
            #------------- params for training --------------
            lr=0.001, wd=0.01, lr_patience=10, lr_warmup=5, lr_scheduler='plateau', lr_epochs=100,# optim parameters
            #------------- params for diffusion --------------
            coeff_ce=0.01,
            ce_only=False,
            num_diffusion_steps=50,
            noise_schedule_type='cosine',
            noise_schedule_args={},
            uniform_noise=False,
            blockwise_timestep=False,
            #------------- params for sampling ---------------
            node_marginal_distribution=None,
            edge_marginal_distribution=None,
            initial_blocksize_distribution=None,
            blocksize_model:PredictBlockProperties=None,
            #------------- additional controls --------------
            combine_training=False, 
            use_relative_blockid=False, # use relative ID can help reduce overfitting, as the absolute id is not meaningful
            use_absolute_blockid=True, # use absolute ID 
            batched_sequential=False,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["blocksize_model"])
        if batched_sequential:
            # Not support yet, can think later. This is a actually possible. 
            blockwise_timestep = False
            assert not combine_training, "batched_sequential and combine_training cannot be both True"
        # ---------------------------------- build model ----------------------------------
        self.to_dense_batch = ToDenseParallelBlocksBatch()
        self.input_encoder = NodeEdgeEncoder(num_node_features, num_edge_features, channels, edge_channels or channels)

        self.input_conditioner = ConditioningSequential(
            BlockIDConditioning(channels, max_num_blocks) if use_absolute_blockid else IdentityConditioning(),
            BlockDegreeConditioning(channels, max_block_degree), 
            TimestepConditioning(channels, channels, blockwise_timestep, only_for_target=combine_training and (not batched_sequential)),            # requires batch.timestep  Bx1
            StructureFeatureConditioning(channels, k_eigval=5, k_eigvec=5) if batched_sequential else IdentityConditioning(),
        )
        self.edge_conditioner = ConditioningSequential(
            RelativeBlockIDEdgeConditioning((edge_channels or channels), edge_channels or channels) if use_relative_blockid else IdentityConditioning(),
            StructureFeatureEdgeConditioning(edge_channels or channels, k_eigval=5, k_eigvec=5) if batched_sequential else IdentityConditioning(),
        ) 
        
        if edge_channels > 0 and n_head > 0:
            self.use_transformer = True 
            self.net = PPGNTransformer(channels, edge_channels, num_layers, n_head, norm=norm, add_transpose=add_transpose, prenorm=prenorm)
        else:
            self.use_transformer = False  
            self.node_edge_combiner = NodeEdgeCombiner(channels, channels, channels)
            self.net = PPGN(channels, num_layers, norm=norm, add_transpose=add_transpose, simplify=False, prenorm=prenorm)

        # need a out node edge combiner 
        self.out_decoder = VirtualBlockOutDecoder(channels, edge_channels or channels, num_node_features, 
                                                  num_edge_features, norm, num_diffusion_steps if use_input else 0) 
        self.real_block_out_decoder = BlockOutDecoder(channels, edge_channels or channels, max_block_size, max_block_degree, 
                                           norm=norm, include_previous_block=True) if combine_training else None

        # ---------------------------------- build diffusion ---------------------------------- 
        self.node_diffusion = DiscreteDiffusion(num_diffusion_steps, num_node_features, noise_schedule_type, noise_schedule_args, ce_only=ce_only)
        self.edge_diffusion = DiscreteDiffusion(num_diffusion_steps, num_edge_features, noise_schedule_type, noise_schedule_args, ce_only=ce_only)
        uniform_node_noise, uniform_edge_noise = torch.ones(num_node_features), torch.ones(num_edge_features)
        # uniform_node_noise[-1], uniform_edge_noise[-1] = 0, 0 # virtual type should have 0 probability
        self.register_buffer('node_noise', node_marginal_distribution if not uniform_noise else uniform_node_noise / uniform_node_noise.sum())
        self.register_buffer('edge_noise', edge_marginal_distribution if not uniform_noise else uniform_edge_noise / uniform_edge_noise.sum())
        self.register_buffer('initial_blocksize_distribution', initial_blocksize_distribution)
        print(f"node_noise: {self.node_noise}, {self.node_noise.shape}")
        print(f"edge_noise: {self.edge_noise}, {self.edge_noise.shape}")

        self.coeff_ce = coeff_ce
        self.num_diffusion_steps = num_diffusion_steps
        self.blockwise_timestep = blockwise_timestep
        self.max_block_size = max_block_size
        self.max_num_blocks = max_num_blocks
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        self.blocksize_model = blocksize_model
        self.combine_training = combine_training
        self.batched_sequential = batched_sequential
        self.lr = lr 
        self.wd = wd 
        self.lr_patience = lr_patience
        self.lr_warmup = lr_warmup
        self.lr_scheduler = lr_scheduler
        self.lr_epochs = lr_epochs

    def forward_representation(self, batch):
        assert hasattr(batch, 'timestep'), "dense batch should have timestep" 
        edge_causal_mask = get_causal_block_mask(batch.nodes_blockid, batch.virtual_node_mask) if not self.batched_sequential else None
        edge_mask = get_edge_mask_from_block_id(batch.nodes_blockid, batch.virtual_node_mask)
        # encode node and edge: batch.nodes and batch.edges can be used for output conditioning in diffuison 
        node, edge = self.input_encoder(batch)
        # node edge input conditioning
        node = self.input_conditioner(node, batch)
        edge = self.edge_conditioner(edge, batch)  # add edge conditioning 
        if self.use_transformer:
            node, edge = self.net(node, edge, edge_mask, edge_causal_mask)
        else:
            edge = self.node_edge_combiner(node, edge)
            edge = self.net(edge, edge_mask, edge_causal_mask)
        return node, edge

    def forward(self, batch):
        ## assume batch is dense batch
        node, edge = self.forward_representation(batch)
        full_node_logit, full_edge_logit = self.out_decoder(node, edge, batch, causal=not self.batched_sequential)
        return full_node_logit, full_edge_logit 

    @torch.no_grad()
    def _get_help_transforms(self, batch):
        if self.batched_sequential:
            target_node_mask, target_edge_mask = get_last_target_block_mask(batch.nodes_blockid, batch.virtual_node_mask, lower_triangular_only=True, causal=False)
            real_node_mask, real_edge_mask = target_node_mask, target_edge_mask
        else:
            target_node_mask, target_edge_mask, real_node_mask, real_edge_mask = get_target_block_mask(batch.nodes_blockid, batch.virtual_node_mask)
        # get node and edge transforms 
        block_size = batch.block_size if self.blockwise_timestep else None 
        target_node_transforms = prepare_targetblock_transforms(target_node_mask, block_size)
        target_edge_transforms = prepare_targetblock_transforms(target_edge_mask, block_size)
        real_node_transforms = prepare_targetblock_transforms(real_node_mask, block_size)
        real_edge_transforms = prepare_targetblock_transforms(real_edge_mask, block_size)
        return target_node_transforms, target_edge_transforms, real_node_transforms, real_edge_transforms
    
    @torch.no_grad()
    def prepare_input(self, batch):
        # get target block mask for node and edge 
        target_node_transforms, target_edge_transforms, real_node_transforms, real_edge_transforms = self._get_help_transforms(batch)
        batch_size, device = target_node_transforms[2].size(0), batch.nodes_blockid.device
        # sample time 
        time = torch.randint(1, self.num_diffusion_steps+1, (batch_size,), device=device) # B,1 or total_num_blocks, 1
        batch.timestep = time
        # get x0 
        batch_n0 = real_node_transforms[0](batch.nodes, flatten_input=False).argmax(dim=-1)
        batch_e0 = real_edge_transforms[0](batch.edges, flatten_input=False).argmax(dim=-1)
        # forward sampling 
        batch_nt = self.node_diffusion.qt_0_sample(batch_n0, time, m=self.node_noise, conditional_mask=real_node_transforms[2]) # B x m_max 
        batch_et = self.edge_diffusion.qt_0_sample(batch_e0, time, m=self.edge_noise, conditional_mask=real_edge_transforms[2]) # B x me_max 
        # go back to full_edges and full_nodes 
        batch.nodes = target_node_transforms[1](batch.nodes, F.one_hot(batch_nt, self.num_node_features).float())
        batch.edges = target_edge_transforms[1](batch.edges, F.one_hot(batch_et, self.num_edge_features).float())

        # assert (batch.nodes[..., -1] == 0).all() and (batch.edges[..., -1] == 0).all(), "Virtual type should not be inside input"
        return batch, batch_n0, batch_e0, batch_nt, batch_et, target_node_transforms, target_edge_transforms
    
    def _shared_step(self, batch, split='train'):
        # get target 
        batch = self.to_dense_batch(batch)
        batch, batch_n0, batch_e0, batch_nt, batch_et, target_node_transforms, target_edge_transforms = self.prepare_input(batch)
        batch_size = batch.nodes.size(0)
        # get prediction
        node, edge = self.forward_representation(batch)
        full_node_logit, full_edge_logit = self.out_decoder(node, edge, batch)
        batch_node_logit = target_node_transforms[0](full_node_logit, flatten_input=False) # B x max_nodes x F
        batch_edge_logit = target_edge_transforms[0](full_edge_logit, flatten_input=False) # B x max_edges x C

        # compute diffusion loss over nodes and edges
        node_loss_dict = self.node_diffusion.compute_loss(batch_node_logit, 
                                            x_t=batch_nt, 
                                            x_0=batch_n0, 
                                            t=batch.timestep, 
                                            m=self.node_noise, 
                                            coeff_ce=self.coeff_ce, 
                                            conditional_mask=target_node_transforms[2]
                                        )
        edge_loss_dict = self.edge_diffusion.compute_loss(batch_edge_logit, 
                                            x_t=batch_et, 
                                            x_0=batch_e0, 
                                            t=batch.timestep, 
                                            m=self.edge_noise, 
                                            coeff_ce=self.coeff_ce, 
                                            conditional_mask=target_edge_transforms[2]
                                        )
        metrics = {f'{split}/node/vlb_nodes': node_loss_dict['vlb_loss']/batch_size, 
                   f'{split}/edge/vlb_edges': edge_loss_dict['vlb_loss']/batch_size,
                   f'{split}/node/ce_nodes':  node_loss_dict['ce_loss']/batch_size,
                   f'{split}/edge/ce_edges':  edge_loss_dict['ce_loss']/batch_size}
        self.log_dict(metrics, on_epoch=True, batch_size=batch_size)
        
        if self.combine_training:
            # compute loss over real blocks for block property prediction 
            size_pred, size_target, degree_pred, degree_target, mask, first_degree_pred, first_degree_target = self.real_block_out_decoder(node, edge, batch, return_only_first_block=False)
            size_loss = F.cross_entropy(size_pred[mask], size_target[mask], reduction='sum')
            degree_loss = F.cross_entropy(degree_pred[mask], degree_target[mask], reduction='sum')
            first_degree_loss = F.cross_entropy(first_degree_pred, first_degree_target, reduction='sum')
            block_loss = size_loss + degree_loss + first_degree_loss
            block_metrics = {f'{split}/block/loss':block_loss / batch_size, 
                             f'{split}/block/size_acc': (size_pred[mask].argmax(dim=-1) == size_target[mask]).float().mean(),
                             f'{split}/block/degree_acc': (degree_pred[mask].argmax(dim=-1) == degree_target[mask]).float().mean(),
                             f'{split}/block/initdeg_acc':(first_degree_pred.argmax(dim=-1) == first_degree_target).float().mean()}
            self.log_dict(block_metrics, on_epoch=True, batch_size=batch_size)
        else:
            block_loss = 0

        # combine all losses together 
        final_loss = (node_loss_dict['loss'] + edge_loss_dict['loss'] + block_loss)/batch_size
        self.log(f"{split}_loss", final_loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        return final_loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, split='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, split='val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, split='test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.lr_warmup,
                                                        num_training_steps=self.lr_epochs,
                                                        min_lr=1e-6,
                                                        min_lr_mode="clamp")
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=0.5,
                                                                patience=self.lr_patience,
                                                                verbose=True)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        
    @torch.no_grad()
    def predict_block_properties(self, dense_batch, pred_init_degree=False):
        self.eval()
        if self.blocksize_model is not None:
            return self.blocksize_model.inference(dense_batch, pred_init_degree=pred_init_degree)
        else:
            assert not self.batched_sequential, "batched_sequential is not supported when blocksize_model is None"
            if pred_init_degree:
                # predict the init block_size, input should contains the batched current block's prediction 
                pred = self.real_block_out_decoder(None, None, dense_batch, return_only_first_block=True)[0] # B x max_block_degree
                dist = F.softmax(pred, dim=-1) # distribution
                next_block_degree = sample_categorical(dist) # B
                return next_block_degree  # B
            
            assert dense_batch.virtual_node_mask.sum() == 0, "Make sure that all blocks are completed and marked as real blocks"
            node, edge = self.forward_representation(dense_batch)
            batch_size = node.size(0)

            # predict next size 
            out = self.real_block_out_decoder(node, edge, dense_batch, return_only_first_block=False)
            size_pred, block_mask = out[0], out[4]
            size_pred = F.softmax(size_pred, dim=-1) # distribution, B x max_block_size x Class

            # get last block, by identify the first 0 in block_size
            block_size = dense_batch.block_size                # B x max_num_blocks
            next_block_idx = block_mask.sum(dim=-1)      # B
            last_block_idx = next_block_idx - 1          # B
            idx1 = torch.arange(batch_size, device=next_block_idx.device)
            dist = size_pred[idx1, last_block_idx]      # B x Class
            next_block_size = sample_categorical(dist) # B

            # predict next degree
            # 1. prepare target block_size 
            blocksize_target = block_size.clone()
            blocksize_target[:, :-1] = blocksize_target[:, 1:]
            blocksize_target[:, -1] = 0
            blocksize_target[idx1, last_block_idx] = next_block_size
            ### next_block_idx can become out of boundary when block_mask if full 
            ### Solution 1: provide the target block size as input to self.out_decoder.
            out = self.real_block_out_decoder(node, edge, dense_batch, return_only_first_block=False, blocksize_target=blocksize_target)
            degree_pred, block_mask = out[2], out[4]
            dist = F.softmax(degree_pred, dim=-1)[idx1, last_block_idx] # B x max_block_degree
            next_block_degree = sample_categorical(dist) # B
            return next_block_size, next_block_degree # B, B 
    
    @torch.no_grad()
    def generate_last_block(self, dense_batch):
        """ 
        Sampling start from the initial noise distribution, then to the target x0.
        This assumes that the last block's degree and size are known.
        """
        assert hasattr(dense_batch, 'nodes_blockid')
        assert hasattr(dense_batch, 'block_size')
        assert hasattr(dense_batch, 'block_degree')
        batch_size, device = dense_batch.nodes.size(0),  dense_batch.nodes.device
        diag_idx = torch.arange(dense_batch.nodes.size(1), device=device)

        last_block_node_mask, last_block_edge_mask = get_last_target_block_mask(dense_batch.nodes_blockid, dense_batch.virtual_node_mask, lower_triangular_only=True, causal=True)
        node_transforms = prepare_targetblock_transforms(last_block_node_mask)
        edge_transforms = prepare_targetblock_transforms(last_block_edge_mask)
        t = torch.full((batch_size,), self.num_diffusion_steps, device=device) # Tmax 

        ### the first step: sample from the true noise distribution
        # get batch nt and et
        batch_nt, batch_et = node_transforms[0](dense_batch.nodes), edge_transforms[0](dense_batch.edges) # B x n_node x F, B x n_edge x C
        n_t = torch.multinomial(self.node_noise, num_samples=batch_nt[...,0].numel(), replacement=True).view(batch_nt[...,0].shape)
        e_t = torch.multinomial(self.edge_noise, num_samples=batch_et[...,0].numel(), replacement=True).view(batch_et[...,0].shape)

        # start from t=T, to t=1
        for i in range(self.num_diffusion_steps):
            # go back to full graph
            dense_batch.nodes = node_transforms[1](dense_batch.nodes, F.one_hot(n_t, self.num_node_features).float())
            dense_batch.edges = edge_transforms[1](dense_batch.edges, F.one_hot(e_t, self.num_edge_features).float())
            dense_batch.timestep = t

            assert (dense_batch.edges[:,diag_idx,diag_idx] == 0).all(), "Diagonal of formatted_batch.edges should be 0"
            # get prediction
            full_node_logit, full_edge_logit = self(dense_batch)
            batch_node_logit = node_transforms[0](full_node_logit, flatten_input=False) # B x max_nodes x F
            batch_edge_logit = edge_transforms[0](full_edge_logit, flatten_input=False) # B x max_edges x C
            n_probt = from_logits_to_prob(batch_node_logit, no_virtual=(i == self.num_diffusion_steps - 1))
            e_probt = from_logits_to_prob(batch_edge_logit, no_virtual=(i == self.num_diffusion_steps - 1))

            # sample from the current marginal distribution
            assert (n_probt.sum(-1) > 0).all() and (e_probt.sum(-1) > 0).all(), f"{i}/{self.num_diffusion_steps}, n_probt and e_probt should be non-negative"
            s = t - 1
            n_t = self.node_diffusion.sample_step(n_probt, n_t, t, s, m=self.node_noise, conditional_mask=node_transforms[2])
            e_t = self.edge_diffusion.sample_step(e_probt, e_t, t, s, m=self.edge_noise, conditional_mask=edge_transforms[2])
            t = s
        ## check whether virtual type is inside final sampling step
        assert (n_t[~node_transforms[2]] < self.num_node_features -1).all(), "Virtual type is inside final sampling step"
        assert (e_t[~edge_transforms[2]] < self.num_edge_features -1).all(), "Virtual type is inside final sampling step"
        dense_batch.nodes = node_transforms[1](dense_batch.nodes, F.one_hot(n_t, self.num_node_features).float())
        dense_batch.edges = edge_transforms[1](dense_batch.edges, F.one_hot(e_t, self.num_edge_features).float())
        dense_batch.virtual_node_mask.fill_(False) # after generation, virtual node mask should be all False
        return dense_batch
    
    @torch.no_grad()
    def generate(self, batch_size): 
        assert (self.blocksize_model is not None) or self.combine_training, "Need blocksize/degree model to generate"  
        initial_size_distribution = self.initial_blocksize_distribution.to(self.device).unsqueeze(0)
        initial_block_size = sample_categorical(initial_size_distribution.repeat(batch_size, 1)) # B
        self.to(self.device).eval()
        if self.blocksize_model is not None:
            self.blocksize_model.to(self.device).eval()

        # build the first input block 
        dense_batch = self.create_init_dense_batch(initial_block_size)
        initial_block_degree = self.predict_block_properties(dense_batch, pred_init_degree=True)
        dense_batch.block_degree = initial_block_degree.unsqueeze(-1)

        #### start the generation loop 
        finished_generation = torch.full_like(initial_block_size, False, dtype=torch.bool)
        generated_max_blocks = initial_block_size.clone()
        i = 0 
        while (not finished_generation.all()) and i<self.max_num_blocks:
            if i > 0:
                dense_batch = add_block_to_dense_batch(dense_batch, next_block_size, next_block_degree)
            # refine the last block 
            dense_batch = self.generate_last_block(dense_batch)
  
            # predict the next block size and block degree, and set flag of finishing 
            next_block_size, next_block_degree = self.predict_block_properties(dense_batch, pred_init_degree=False)
            finished_mask = (next_block_size == 0)
            generated_max_blocks[(~finished_generation) & finished_mask] = i 
            finished_generation = finished_generation | finished_mask 

            # if finished, set next block size and degree to 0
            next_block_size[finished_generation] = 0
            next_block_degree[finished_generation] = 0
            i += 1
           
        # process generated graph by masking out any part larger than generated_max_blocks
        assert dense_batch.nodes_blockid.dim() == 2
        nodes_mask = (dense_batch.nodes_blockid <= generated_max_blocks.unsqueeze(-1)) & (dense_batch.nodes_blockid >=0) # B x N_max 
        edges_mask = nodes_mask.unsqueeze(1) & nodes_mask.unsqueeze(-1) # B x N_max x N_max
        dense_batch.nodes[~nodes_mask] = 0
        dense_batch.edges[~edges_mask] = 0
        dense_batch.nodes_blockid[~nodes_mask] = -1
        return dense_batch # the generated batch of graphs

    def create_init_dense_batch(self, initial_block_size):
        batch_size, Nmax, device = initial_block_size.size(0), initial_block_size.max(), initial_block_size.device
        node_mask = torch.arange(Nmax, device=device) < initial_block_size.unsqueeze(-1)
        nodes = torch.zeros(batch_size, Nmax,       self.num_node_features, dtype=torch.float, device=device)
        edges = torch.zeros(batch_size, Nmax, Nmax, self.num_edge_features, dtype=torch.float, device=device)
        block_id = torch.zeros(batch_size, Nmax, dtype=torch.long, device=device)
        block_id[~node_mask] = -1
        dense_batch = Batch(
            nodes=nodes,            # B x Nmax x F
            edges=edges,            # B x Nmax x Nmax x C
            nodes_blockid=block_id, # B x Nmax,
            virtual_node_mask=node_mask, # last node block is masked as virtual
            block_size=initial_block_size.view(-1,1),  # B x 1
        )
        return dense_batch
    
    @torch.no_grad()
    def compute_log_probability(self, batch):

        # Compute sum of log probabilities of predicting next block size + the log probability of initial block size

        # Compute sum of log probabilities of predicting next block for all diffusion steps 
        pass 

import torch.nn.functional as F    
def add_block_to_dense_batch(dense_batch, added_block_size, added_block_degree):
    """
    added_block_size  : B x 1
    """
    new_sizes = (dense_batch.nodes_blockid>=0).sum(-1) + added_block_size.squeeze() # B
    new_n_max = new_sizes.max()
    delta_n   = new_n_max - dense_batch.nodes.size(1)

    new_node_mask = torch.arange(new_n_max, device=dense_batch.nodes_blockid.device) < new_sizes.unsqueeze(-1) # B x N_max
    new_block_id = F.pad(dense_batch.nodes_blockid, (0,delta_n), value=-1)
    next_id = (dense_batch.nodes_blockid.max(dim=-1, keepdim=True)[0] + 1).repeat(1, new_n_max) # B x N_max
    new_mask = new_node_mask & (new_block_id<0)
    new_block_id[new_mask] = next_id[new_mask]

    dense_batch.nodes = F.pad(dense_batch.nodes, (0,0,0,delta_n), value=0.) 
    dense_batch.edges = F.pad(dense_batch.edges, (0,0,0,delta_n,0,delta_n), value=0.)
    dense_batch.nodes_blockid     = new_block_id 
    dense_batch.virtual_node_mask = new_mask # completed samples doesn't have virtual node 
    # (new_block_id == new_block_id.max(dim=-1, keepdim=True)[0])  # last node block is masked as virtual
    # TODO: not all last blocks are masked in generation! Some samples are finished generation, and the last block shoudn't be changed! 
    dense_batch.block_size        = torch.cat([dense_batch.block_size, added_block_size.view(-1,1)], dim=-1)
    dense_batch.block_degree      = torch.cat([dense_batch.block_degree, added_block_degree.view(-1,1)], dim=-1)
    return dense_batch

def from_logits_to_prob(logits, no_virtual=False, temperature=1.0):
    if no_virtual:
        # logits += 1e-30
        logits[..., -1] = -np.inf 
    prob = F.softmax(logits / temperature, dim=-1)
    return prob
