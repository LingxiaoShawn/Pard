# Pard: Autoregressive + Diffusion 

Official pytorch source code for 

> [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://arxiv.org/abs/2402.03687)   
> Lingxiao Zhao, Xueying Ding, Leman Akoglu


## About

Pard **combines** <ins>autoregressive approach</ins> and <ins>diffusion model</ins> together. Instead of working with the full joint distribution of nodes and edges directly (like other Diffusion model, e.g. [DiGress](https://arxiv.org/pdf/2209.14734) ), pard decomposes the joint distribution into product of many conditional distributions (for blocks), where each conditional distribution is modeled by diffusion model. As each conditional distribution's internal dependency is a lot simpler than the full joint distribution, it can be captured easily with significantly fewer diffusion steps and no extra feature is needed for symmetry breaking. What is more, different from autoregressive approach that suffers from arbitary ordering issue without exchangable probability, pard is a **permutation-invariant** model. 


### Highlight 
1. Pard beats both autoregressive approach and diffusion model in both molecular and non-molecular datasets. 
2. Comparing to diffusion model, Pard uses significantly less number of diffusion steps, and does NOT need any extra feature. 
3. Similar to GPT whose all autoregressive steps are trained in parallel, Pard trains all blocks' conditional distribution in parallel with a shared diffusion model.  
4. Pard's framework of AR + Diffusion is general, which can be used for other modality like text and image. Pard is the **first** AR + Diffusion model.


## Installation
See [setup.sh](https://github.com/LingxiaoShawn/Pard/blob/main/setup.sh) for installing the package. Most packages can be installed with newest version. If you meet a problem, submit an issue and we will try to update the coode for the newest environment.  


## Code structure 

#### Part1 (Peripheral): 
* `pard/analysis` contains functions for evaluation, with molecular specific evaluations and non-molecular MMD evaluations. This part is modified from [DiGress](https://github.com/cvignac/DiGress). 
* `pard/dataset` contains many datasets formatted in `torch_geometric.datasets` format. Some are adapted from  [DiGress](https://github.com/cvignac/DiGress). 
* `pard/configs` contains yaml config files for datasets shown in paper. We have not checked exactly, but most numbers in our paper should align with the config file in this folder. 


#### Part2 (Core):
* **`pard/parallel`** contains the main part of codes. The model architecture is inside `pard/parallel/network.py`, and the pard's training and generation code inside `pard/parallel/task.py`. Many data transformation needed is inside `pard/parallel/transform.py`. Notice that the code inside `pard/parallel/task.py` supports both parallel training and sequential batch based training. We will introduce it inside next section. 
* `pard/main.py` contains the entry for training the pard.
* `pard/eval.py` contains the entry for generation and evaluation. 


## Usage

### Train 

#### A. Train with all conditional blocks in parallel. 
> Similar to GPT which predicts next token for all tokens in parallel, this mode predict next block for all conditional blocks in parallel. To achieve this, for a graph with $n$ nodes, we augment the graph with empty nodes and edges to the size $2n$, such that the second part with nodes ranging from $n \sim 2n$ contains the predicted blocks. See [augment_with_virtual_blocks](https://github.com/LingxiaoShawn/Pard/blob/ef03750b84d3d2f5ad913d299e2f2277450419c0/pard/parallel/transform.py#L123) for padding the input graph fron $n$ nodes to $2n$ nodes. Apart from augmenting the graph, we also need to make sure there is no information leakage when predicting the $i+1$ block from the previous $1\sim i$ blocks. This is achieved with specifically designed *causal graph transformer*. See section 4.2 in [paper](https://arxiv.org/abs/2402.03687). 
   * Train local diffusion model 
      ```python
      python main.py device 0 dataset qm9 task local_denoising diffusion.num_steps 20
      ```
   * Train predicting next block's size model 
      ```python
      python main.py device 0 dataset qm9 task block_prediction diffusion.num_steps 20
      ```

   * (If you want train only a single model) Train local diffusion and predicting next block's size together 

      ```python
      python main.py device 0 dataset qm9 task local_denoising diffusion.num_steps 20 diffusion.combine_training True
      ```

#### B. Train with sequentially batched blocks. See [sequential_setting](https://github.com/LingxiaoShawn/Pard/blob/ef03750b84d3d2f5ad913d299e2f2277450419c0/pard/parallel/transform.py#L158) for the code.  
> While training all blocks in parallel can be more efficient in memory and passing data, thanks to <ins>shared</ins> representations of nodes and edges for all blocks' diffusion prediction. The parallel training loses certain model expressiveness (PPGN with restricted matrix multiplication for preventing information leakage will be less powerful than 3-WL), and extra features like eigenvectors are not possible to be used without information leakage. Another way to train is to view each block of a graph as a unique graph, and batch all blocks into a gaint graph. We provide the support in [sequential_setting](https://github.com/LingxiaoShawn/Pard/blob/ef03750b84d3d2f5ad913d299e2f2277450419c0/pard/parallel/transform.py#L158). However, this will greatly increase the memory cost such that the batch size need to be reduced, hence training time will be longer when considering training with same number of epochs. The goodness is that certain extra feature like eigenvector can be easily added to further break symmetry in diffusion, which makes predicting next block task a lot easiler for symmetry-rich graphs like grids. 
   * Train local diffusion model 
      ```python
      python main.py device 0 dataset qm9 task local_denoising diffusion.num_steps 20 model.batched_sequential True train.batch_size 32 ## remember to tune the batch_size smaller to fit the memory
      ```
   * Train predict next block's size model 
      ```python
      python main.py device 0 dataset qm9 task block_prediction diffusion.num_steps 20 model.batched_sequential True train.batch_size 32 ## remember to tune the batch_size smaller to fit the memory
      ```
   * (If you want train only a single model) Train local diffusion and predicting next block's size together 

      ```python
      python main.py device 0 dataset qm9 task local_denoising diffusion.num_steps 20 diffusion.combine_training True model.batched_sequential True train.batch_size 32
      ```  

### Generation and Evaluation 

``` python 
from eval import eval_model 

device = 0
dataset = 'qm9' 
batch_size = 1024  # can be larger than training batch_size as inference uses less memory
train_max_hops = 3 # this should align with training config, diffusion.max_hop

# Provide the checkpoint dir, see 'checkpoint_callback' in main.py
blocksize_model_dir = '...' # if you train two tasks together, let it be blocksize_model_dir=None
diffusion_model_dir = '...' 

eval_mode = 'best' # 'all' to evaluate all checkpoints, 'best' to evaluate the one with best validation performance, 'latest' to evaluate the last checkpoint

eval_model(device, dataset, diffusion_model_dir, blocksize_model_dir, eval_mode, batch_size=batch_size, train_max_hops=train_max_hops)
```

## Diffusion

We use the most basic discrete-time diffusion method ([pard/diffusion.py](https://github.com/LingxiaoShawn/Pard/blob/main/pard/diffusion.py)) from [USD3](https://github.com/LingxiaoShawn/USD3). The diffusion code ([pard/diffusion.py](https://github.com/LingxiaoShawn/Pard/blob/main/pard/diffusion.py)) notation follows the USD3 paper exactly. The USD3 contains more sophisticated discrete diffusion, with improvement in both discrete-time and continuous-time case. We suggest people who want to use discrete diffusion to check out the code and the [paper](https://arxiv.org/pdf/2402.03701.pdf).



## Citation 
If you use this codebase, or otherwise found our work valuable, please cite:

```
@article{zhao2024pard,
  title={Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation},
  author={Zhao, Lingxiao and Ding, Xueying and Akoglu, Leman},
  journal={arXiv preprint arXiv:2402.03687},
  year={2024}
}

@article{zhao2024improving,
  title={Improving and Unifying Discrete\&Continuous-time Discrete Denoising Diffusion},
  author={Zhao, Lingxiao and Ding, Xueying and Yu, Lijun and Akoglu, Leman},
  journal={arXiv preprint arXiv:2402.03701},
  year={2024}
}
```






