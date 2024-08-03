from yacs.config import CfgNode as CN

def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'zinc'
    cfg.task = 'local_denoising'
    # Additional num of worker for data loading
    cfg.num_workers = 16
    # Cuda device number, used for machine with multiple gpus
    cfg.device = None
    # Additional string add to logging 
    cfg.handtune = ''
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # version 
    cfg.version = 'final'
    cfg.eval_only = False 
    cfg.eval_all_checkpoints = True

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    cfg.train.precision = '32-true' # default training precision 
    cfg.train.resume = False # try to resume from the last checkpoint
    cfg.train.resume_mode = 'best' # best or last
    # Total graph mini-batch size
    cfg.train.batch_size = 512
    # Maximal number of epochs
    cfg.train.epochs = 100
    # Number of runs with random init 
    cfg.train.runs = 3
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 20
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    cfg.train.lr_warmup = 5
    cfg.train.lr_scheduler = 'cosine'  # cosine, plateau
    # L2 regularization, weight decay
    cfg.train.wd = 0.01
    # Dropout rate
    cfg.train.dropout = 0.
    
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    cfg.model.arch = 'PPGN'                        # [main param] model architecture, 'SWL'
    cfg.model.hidden_size = 128                    # hidden size of the model
    cfg.model.num_layers = 6                       # [main param] number of layers
    cfg.model.num_heads = 4                        # [main param] number of attention heads
    cfg.model.norm = 'bn'
    cfg.model.add_transpose = True
    cfg.model.prenorm = True 
    cfg.model.num_heads = 8
    cfg.model.edge_hidden = 32                     # 0 means use PPGN only
    cfg.model.transformer_only = False             # whether only use transformer, when edge_hidden > 0, this will work, can increase edge_hidden abit. 
    cfg.model.extra_feature = False                # whether to use extra structure feature (cycle, eigen)
    cfg.model.encode_block = True
    cfg.model.input_residual = True
    cfg.model.use_relative_blockid = False
    cfg.model.use_absolute_blockid = True
    cfg.model.batched_sequential = False # if False use parallel training, if true use batched sequential of partial graphs for each graph
    
    # ------------------------------------------------------------------------ #
    # Diffusion options
    # ------------------------------------------------------------------------ #
    cfg.diffusion = CN()
    cfg.diffusion.max_hops = 3                     # [main param] number of hops for generating blocks
    cfg.diffusion.num_steps = 20                   # [main param] number of diffusion steps
    cfg.diffusion.noise_schedule_type = 'cosine'   # [main param] type of noise schedule
    cfg.diffusion.ce_only = False                  # [main param] whether to use only cross entropy loss
    cfg.diffusion.ce_coeff = 0.01                  # [main param] coefficient of the cross entropy loss P(x_0|x_t)
    cfg.diffusion.uniform_noise = False            # [main param] whether to use uniform noise schedule
    cfg.diffusion.marginal_temperature = 1.0       # [main param] temperature of the marginal distribution
    cfg.diffusion.allow_online_evaluation = False  # [main param] whether to allow online evaluation during training 
    cfg.diffusion.no_virtual_type = False
    cfg.diffusion.blockwise_time = False 
    cfg.diffusion.combine_training = False 
    cfg.diffusion.num_node_virtual_types = 1             # number of virtual types for node
    cfg.diffusion.num_edge_virtual_types = 1             # number of virtual types for edge
    
    return cfg
    
import os 
import argparse
# Principle means that if an option is defined in a YACS config object, 
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining, 
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.

def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, 
                         help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg 
    cfg = cfg.clone()
    
    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line 
    cfg.merge_from_list(args.opts)
       
    return cfg

"""
    Global variable
"""
cfg = set_cfg(CN())