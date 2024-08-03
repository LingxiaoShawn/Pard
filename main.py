import warnings
warnings.filterwarnings('ignore')
import torch
import lightning as L
import os, sys
from tqdm import tqdm
from pard.config import cfg, update_cfg
from pard.dataset import DATA_INFO
from pard.parallel.transform import ToParallelBlocks
from pard.parallel.transform import ToOneHot
from pard.parallel.task import PredictBlockProperties, AutoregressiveDiffusion
import pard.parallel.utils as parallel_utils
from pard.utils import find_checkpoint_with_lowest_val_loss


# --------------------------------------- input --------------------------------------------
dataset_name = sys.argv[sys.argv.index('dataset')+1]
cfg.merge_from_file(f'pard/configs/{dataset_name}.yaml')
cfg = update_cfg(cfg)
cfg.dataset = dataset_name.split('-')[0]
if isinstance(cfg.device, int):
    torch.cuda.set_device(cfg.device)
torch.set_num_threads(cfg.num_workers)
assert cfg.task in ['block_prediction', 'local_denoising']

batched_sequential = cfg.model.batched_sequential
# ------------------------------------- parameters ------------------------------------------
data_info_dict = DATA_INFO[cfg.dataset]
num_node_features = data_info_dict['num_node_features']
num_edge_features = data_info_dict['num_edge_features']
start_edge_type   = data_info_dict['start_edge_type']
atom_decoder = data_info_dict.get('atom_decoder', None)
metric_class = data_info_dict.get('metric_class', None)

# --------------------------------------- data --------------------------------------------
print(f'Loading {cfg.dataset} dataset...')
from torch_geometric.transforms import Compose
one_hot = ToOneHot(num_node_features, num_edge_features, virtual_node_type=cfg.diffusion.num_node_virtual_types, 
                   virtual_edge_type=cfg.diffusion.num_edge_virtual_types, has_zero_edgetype=start_edge_type==0)
to_parallel_blocks = ToParallelBlocks(max_hops=cfg.diffusion.max_hops, 
                                      add_virtual_blocks= (cfg.task=='local_denoising') and not batched_sequential, 
                                      to_batched_sequential=batched_sequential)
train_transform = Compose([one_hot, to_parallel_blocks])
datasets = {
    split:data_info_dict['class'](**(
                data_info_dict['default_args'] |
                {'split':split} |
                {'transform': train_transform}
            )) 
    for split in ['train', 'val', 'test']
}

# do transform offline to save time, problem: not good for distributed setting, since each process needs to do this
datasets = {
    split: [d for d in tqdm(datasets[split], desc=f'Transforming {split} dataset')]
    for split in ['train', 'val', 'test']
}

train_vali = datasets['train'] + datasets['val']

#### problem: computing all these dataset statistics is time-consuming for large datasets
node_dist, edge_dist = parallel_utils.get_node_edge_marginal_distribution(train_vali)
init_size_dist, init_degree_dist, list_num_blocks = parallel_utils.get_init_block_size_degree_marginal_distrbution(train_vali)
max_num_blocks, mean_num_blocks = max(list_num_blocks), sum(list_num_blocks)/len(list_num_blocks)
max_block_size, max_block_degree = len(init_size_dist), len(init_degree_dist)
print(f'Number of blocks: {max_num_blocks}, max block size: {max_block_size}, max block degree: {max_block_degree}')
print(f'Average number of blocks in training set: {mean_num_blocks}, Average total diffusion steps: {mean_num_blocks*cfg.diffusion.num_steps}')

# --------------------------------------- loader --------------------------------------------
from torch_geometric.loader import DataLoader
loaders = {
    split:  DataLoader(
                datasets[split], 
                batch_size=cfg.train.batch_size if split=='train' else cfg.train.batch_size * 4, 
                shuffle=True if split=='train' else False, 
                num_workers=cfg.num_workers,
                pin_memory=True,          # turn off if needs more memory on GPUs, but may slow down training without lower GPU usage 
                persistent_workers=True,  # turn on to avoid kill loaders after each epoch 
            )
    for split in ['train', 'val', 'test']
}
# --------------------------------------- model --------------------------------------------
print(f'Building model...')
if cfg.task == 'block_prediction':
    model = PredictBlockProperties(
        #------------- params for models -----------------
        one_hot.num_node_classes, 
        one_hot.num_edge_classes, 
        max_num_blocks + 3, # add 3 for tolerance of nn.embedding
        max_block_size,
        max_block_degree,
        channels=cfg.model.hidden_size, 
        num_layers=cfg.model.num_layers, 
        norm=cfg.model.norm, 
        add_transpose=cfg.model.add_transpose,   # PPGN parameters
        prenorm=cfg.model.prenorm,
        edge_channels=cfg.model.edge_hidden, 
        n_head=cfg.model.num_heads,              # PPGNTransformer's additional parameters 
        transformer_only=cfg.model.transformer_only, 
        #------------- params for training -----------------
        lr=cfg.train.lr, 
        wd=cfg.train.wd, 
        lr_patience=cfg.train.lr_patience,
        lr_warmup=cfg.train.lr_warmup,
        lr_scheduler=cfg.train.lr_scheduler,
        lr_epochs=cfg.train.epochs,
        use_relative_blockid=cfg.model.use_relative_blockid, 
        use_absolute_blockid=cfg.model.use_absolute_blockid,
        batched_sequential=batched_sequential,
    )
elif cfg.task == 'local_denoising':
    model = AutoregressiveDiffusion(
        #------------- params for models -----------------
        one_hot.num_node_classes, 
        one_hot.num_edge_classes, 
        max_num_blocks + 3, # add 5 for tolerance of nn.embedding
        max_block_size,
        max_block_degree,
        channels=cfg.model.hidden_size, 
        num_layers=cfg.model.num_layers, 
        norm=cfg.model.norm, 
        add_transpose=cfg.model.add_transpose,   # PPGN parameters
        prenorm=cfg.model.prenorm,
        edge_channels=cfg.model.edge_hidden, 
        n_head=cfg.model.num_heads,              # PPGNTransformer's additional parameters 
        transformer_only=cfg.model.transformer_only, 
        use_input=cfg.model.input_residual,      # AutoregressiveDiffusion's additional parameters
        #------------- params for training -----------------
        lr=cfg.train.lr, 
        wd=cfg.train.wd, 
        lr_patience=cfg.train.lr_patience,
        lr_warmup=cfg.train.lr_warmup,
        lr_scheduler=cfg.train.lr_scheduler,
        lr_epochs=cfg.train.epochs,
        #------------- params for diffusion --------------
        coeff_ce=cfg.diffusion.ce_coeff,
        ce_only=cfg.diffusion.ce_only,
        num_diffusion_steps=cfg.diffusion.num_steps,
        noise_schedule_type=cfg.diffusion.noise_schedule_type,
        noise_schedule_args={},
        uniform_noise=cfg.diffusion.uniform_noise,
        blockwise_timestep=cfg.diffusion.blockwise_time,
        #------------- params for sampling ---------------
        node_marginal_distribution=node_dist,
        edge_marginal_distribution=edge_dist,
        initial_blocksize_distribution=init_size_dist,
        blocksize_model=None,
        combine_training=cfg.diffusion.combine_training,
        use_relative_blockid=cfg.model.use_relative_blockid,
        use_absolute_blockid=cfg.model.use_absolute_blockid,
        batched_sequential=batched_sequential,
    )
else:
    raise NotImplementedError

def initialize_weights(m, gain=1.0):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
from functools import partial
model.apply(partial(initialize_weights, gain=1.0 if cfg.model.norm == 'ln' else 0.5))

# --------------------------------------- trainer --------------------------------------------
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

print(f'Building trainer...')
if cfg.model.add_transpose:
    cfg.handtune += 'ppgnTrans'
if batched_sequential:
    cfg.handtune += '-BatchedSeq'
else:
    cfg.handtune += '-Parallel'
model_name = f'{cfg.dataset}.{cfg.diffusion.max_hops}hops.{cfg.diffusion.num_node_virtual_types}-{cfg.diffusion.num_edge_virtual_types}typeadded.{cfg.handtune}'+\
             f'.BlockID{int(cfg.model.use_absolute_blockid)}{int(cfg.model.use_relative_blockid)}.{cfg.model.norm}'+\
             f'.PreNorm={int(cfg.model.prenorm)}.H{cfg.model.hidden_size}.E{cfg.model.edge_hidden}.L{cfg.model.num_layers}-lr{cfg.train.lr}.{cfg.train.lr_scheduler}'
diffusion_name = f'-ires{int(cfg.model.input_residual)}.blocktime{int(cfg.diffusion.blockwise_time)}.uni_noise{int(cfg.diffusion.uniform_noise)}'+\
                 f'.T{cfg.diffusion.num_steps}.{cfg.diffusion.noise_schedule_type}'+\
                 f'.vlb{int(not cfg.diffusion.ce_only)}.ce{int(cfg.diffusion.ce_only)+cfg.diffusion.ce_coeff}.combine={cfg.diffusion.combine_training}'
if cfg.task == 'local_denoising':
    model_name = model_name + diffusion_name
if cfg.model.transformer_only:
    model_name = 'TF-' + model_name
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.task}/{model_name}", 
                                      monitor="val_loss", 
                                      save_top_k=5, 
                                      mode='min', 
                                      filename='{epoch}-{val_loss:.3f}',
                                      save_last=True)
# check whether resume training
resume_path = None
if cfg.train.resume:
    best_checkpoint_path = find_checkpoint_with_lowest_val_loss(checkpoint_callback.dirpath)
    last_checkpoint_path = os.path.join(checkpoint_callback.dirpath, 'last.ckpt') 
    if os.path.exists(best_checkpoint_path) and cfg.train.resume_mode == 'best':
        print(f'Resume training from {best_checkpoint_path}...')
        resume_path = best_checkpoint_path
    elif os.path.exists(last_checkpoint_path) and cfg.train.resume_mode == 'last':
        print(f'Resume training from {last_checkpoint_path}...')
        resume_path = last_checkpoint_path
    else:
        print(f'No checkpoint found at {last_checkpoint_path}...')

if not cfg.eval_only:
    wb_logger = WandbLogger(project=f'ParallelDiffusion-{cfg.task}', name=f'{model_name}', log_model=False, config=cfg)
else:
    wb_logger = None
tb_logger = TensorBoardLogger('tb', name=f'{cfg.task}.{model_name}')

trainer = L.Trainer(
    default_root_dir=f'exps/{cfg.task}/{model_name}',
    devices=[cfg.device] if isinstance(cfg.device, int) else cfg.device,
    max_epochs=cfg.train.epochs, 
    callbacks=[checkpoint_callback, lr_monitor],
    logger=[tb_logger, wb_logger],
    precision=cfg.train.precision, # use for grid for saving memory. 
)
if not cfg.eval_only:
    print(f'Start training...')
    trainer.fit(model, loaders['train'], loaders['val'], ckpt_path=resume_path)

    print(f'Start test...')
    trainer.test(model, loaders['test'], ckpt_path='best') 
