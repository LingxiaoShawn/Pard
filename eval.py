import warnings
warnings.filterwarnings('ignore')
import os 
from pard.parallel.utils import find_checkpoint_with_lowest_val_loss
from pard.parallel.task import AutoregressiveDiffusion, PredictBlockProperties
from pard.dataset import DATA_INFO
from torch_geometric.loader import DataLoader
from pard.analysis.spectre_utils import SpectreSamplingMetrics
from pard.analysis.rdkit_functions import BasicMolecularMetrics
from pard.utils import from_batch_onehot_to_list, check_block_id_train_vs_generation
from moses.metrics.metrics import get_all_metrics
import logging
import numpy as np
import torch

torch.set_num_threads(20)

def eval_model(device, dataset, diffusion_model_dir, blocksize_model_dir=None, eval_mode='best', batch_size=128, train_max_hops=3):
    assert eval_mode in ['best', 'last', 'all']
    logging.basicConfig(filename=os.path.join(diffusion_model_dir, 'generation_metric.log'), encoding='utf-8', level=logging.DEBUG)
    ### load dataset
    data_info_dict = DATA_INFO[dataset]
    atom_decoder = data_info_dict.get('atom_decoder', None)
    metric_class = data_info_dict.get('metric_class', None)
    original_datasets = {split:data_info_dict['class'](**(data_info_dict['default_args'] | {'split':split})) for split in ['train', 'val', 'test']}

    ### build metrics 
    if atom_decoder is None:
        original_loaders = {
            split:  DataLoader(
                        original_datasets[split], 
                        batch_size=batch_size, 
                        shuffle=True if split=='train' else False, 
                        num_workers=12,
                    )
            for split in ['train', 'val', 'test']
        }
        if metric_class is not None:
            metric = metric_class(original_loaders)
        else:
            metric = SpectreSamplingMetrics(original_loaders)
    else:
        train_smiles = original_datasets['train'].get_smiles(eval=False) if dataset in ['qm9', 'zinc250k'] else None 
        test_smiles = original_datasets['test'].get_smiles(eval=False) if dataset in ['qm9', 'zinc250k'] else None
        metric = BasicMolecularMetrics(atom_decoder, train_smiles) 

    ### load blocksize model
    combine_training = True 
    blocksize_model = None
    if blocksize_model_dir is not None:
        combine_training = False
        blocksize_model_path = find_checkpoint_with_lowest_val_loss(blocksize_model_dir)
        blocksize_model = PredictBlockProperties.load_from_checkpoint(blocksize_model_path, map_location=f'cuda:{device}')

    ### load diffusion model
    if eval_mode == 'all':
        files_list = os.listdir(diffusion_model_dir)
        diffusion_model_paths = [os.path.join(diffusion_model_dir, file) for file in files_list if file.endswith('.ckpt')] 
    elif eval_mode == 'best':
        diffusion_model_paths = [find_checkpoint_with_lowest_val_loss(diffusion_model_dir)]
    else:
        diffusion_model_paths = [os.path.join(diffusion_model_dir, 'last.ckpt')]

    for diffusion_model_path in diffusion_model_paths:
        diffusion_model = AutoregressiveDiffusion.load_from_checkpoint(diffusion_model_path, map_location=f'cuda:{device}')
        diffusion_model.combine_training = combine_training
        diffusion_model.blocksize_model = blocksize_model
        
        logging.info('='*100)
        logging.info('ckpt: ' + str(diffusion_model_path.split('/')[-1]) )
        # print(blocksize_model.batched_sequential, diffusion_model.batched_sequential)
        print('Generating ...')
        eval_batch_size = batch_size
        num_eval_samples = data_info_dict.get('num_eval_generation', len(original_datasets['test']))
        if eval_batch_size > num_eval_samples:
            eval_batch_size = num_eval_samples
        dense_graph_list = []
        block_id_same_with_training = []

        while num_eval_samples > 0:
            try:
                generated_batch = diffusion_model.generate(batch_size=eval_batch_size)
                success = True
            except Exception as e:
                print(f"Error {e} in sampling, retrying...")
                success = False
                continue
            if success:
                generated_batch = generated_batch.cpu()
                dense_graph_list.extend(from_batch_onehot_to_list(generated_batch.nodes, generated_batch.edges))
                num_eval_samples -= eval_batch_size
                # check block id same with training or not 
                block_id_same_with_training.extend(check_block_id_train_vs_generation(generated_batch.nodes, 
                                                                                      generated_batch.edges,
                                                                                      generated_batch.nodes_blockid,
                                                                                      train_max_hops=train_max_hops))
        logging.info(f'Percentage of graphs that have the same generation block path as training block path: 
                      {100*sum(block_id_same_with_training) / len(block_id_same_with_training)} %')
        print('Evaluating ...')
        ### evaluate 
        if atom_decoder is None:
            result = metric(dense_graph_list)
        else:
            validity_dict, dic, unique_smiles, all_smiles = metric(dense_graph_list)
            # save generated smiles
            np.save(os.path.join(diffusion_model_dir, 'generated_smiles.npy'), np.array(all_smiles))

            result = [validity_dict, dic]
            scores = get_all_metrics(gen=all_smiles, k=None, test=test_smiles, train=train_smiles)
            logging.info('-'*50 )
            logging.info(str(scores))
            logging.info('-'*50 )
        logging.info(str(result))



if __name__ == '__main__':


    # device = 4
    # batch_size = 128
    # dataset = 'planar'
    # # diffusion_model_dir = 'checkpoints/local_denoising/planar.1hops.10edgevirtypeppgnTrans.relID1.ln.PreNorm=1.H256.E32.L10-lr0.0005.cosine-ires1.blocktime0.uni_noise1.T50.cosine.vlb0.ce1.1.combine=False/'
    # # blocksize_model_dir = 'checkpoints/block_prediction/planar.1hops.10edgevirtypeppgnTrans.relID1.ln.PreNorm=1.H256.E32.L10-lr0.0005.cosine/'

    # blocksize_model_dir = 'checkpoints/block_prediction/planar.1hops.ppgnTrans.BlockID01.ln.PreNorm=1.H256.E32.L10-lr0.0001.plateau/'
    # diffusion_model_dir = 'checkpoints/local_denoising/planar.1hops.ppgnTrans-BatchedSeq.BlockID01.ln.PreNorm=1.H256.E32.L10-lr0.0002.cosine-ires1.blocktime0.uni_noise1.T50.cosine.vlb1.ce0.1.combine=False/'

    device = 5
    dataset = 'qm9'
    train_max_hops = 3
    batch_size = 1024
    blocksize_model_dir = 'checkpoints/block_prediction/qm9.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine'
    diffusion_model_dir = 'checkpoints/local_denoising/qm9.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine-ires1.blocktime0.uni_noise1.T100.cosine.vlb1.ce0.1.combine=False'

    # device = 0
    # dataset = 'zinc250k'
    # batch_size = 1024
    # blocksize_model_dir = 'checkpoints/block_prediction/zinc250k.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine'
    # diffusion_model_dir = 'checkpoints/local_denoising/zinc250k.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine-ires1.blocktime0.uni_noise1.T50.cosine.vlb1.ce0.1.combine=False/'

    # device = 4
    # dataset = 'moses'
    # batch_size = 2048 # 10 passes
    # blocksize_model_dir = 'checkpoints/block_prediction/moses.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0004.cosine/' 
    # # diffusion_model_dir = 'checkpoints/local_denoising/moses.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0002.plateau-ires1.blocktime0.uni_noise1.T50.cosine.vlb1.ce0.1.combine=True/'
    # diffusion_model_dir = 'checkpoints/local_denoising/moses.3hops.ppgnTrans-Parallel.BlockID11.bn.PreNorm=1.H256.E64.L8-lr0.0002.cosine-ires1.blocktime0.uni_noise1.T50.cosine.vlb1.ce0.1.combine=False.resume/'

    eval_mode = 'best'
    eval_model(device, dataset, diffusion_model_dir, blocksize_model_dir, eval_mode, batch_size=batch_size, train_max_hops=train_max_hops)