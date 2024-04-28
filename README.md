# Pard

Official pytorch source code for 

> [Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation](https://arxiv.org/abs/2402.03687)   
> Lingxiao Zhao, Xueying Ding, Leman Akoglu


## About

Pard **combines** <ins>autoregressive approach</ins> and <ins>diffusion model</ins> together. Instead of working with the full joint distribution of nodes and edges directly (like other Diffusion model, e.g. [DiGress](https://arxiv.org/pdf/2209.14734) ), pard decomposes the joint distribution into product of many conditional distributions (for blocks), where each conditional distribution is modeled by diffusion model. As each conditional distribution's internal dependency is a lot simpler than the full joint distribution, it can be captured easily with significantly fewer diffusion steps and no extra feature is needed for symmetry breaking. What is more, different from autoregressive approach that suffers from arbitary ordering issue without exchangable probability, pard is a **permutation-invariant** model. 


#### Highlight 
1. Pard beats both autoregressive approach and diffusion model in both molecular and non-molecular datasets. 
2. Comparing to diffusion model, Pard uses significantly less number of diffusion steps, and does NOT need any extra feature. 
3. Similar to GPT whose all autoregressive steps are trained in parallel, Pard trains all blocks' conditional distribution in parallel with a shared diffusion model.  
4. Pard's framework of AR + Diffusion is general, which can be used for other modality like text and image. 


## Installation
See [setup.sh](https://github.com/LingxiaoShawn/Pard/blob/main/setup.sh) for installing the package. Most packages can be installed with newest version. If you meet a problem, submit an issue and we will try to update the coode for the newest environment.  


## Code structure 

* `pard/analysis` contains functions for evaluation, with molecular specific evaluations and non-molecular MMD evaluations. This part is modified from [DiGress](https://github.com/cvignac/DiGress). 
* `pard/dataset` contains many datasets formatted in `torch_geometric.datasets` format. Some are adapted from  [DiGress](https://github.com/cvignac/DiGress). 
* `pard/configs` contains yaml config files for datasets shown in paper. We haven't check fully, but most numbers in our paper should align with the config file in this folder. 

## Usage


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






