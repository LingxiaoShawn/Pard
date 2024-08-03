# create conda env
ENV=pard
CUDA=12.1
cuda=cu121

# ### python 3.8 is needed for molsets if you want to install molsets with pip
conda create -n $ENV python=3.10 -y 
conda activate $ENV

# install pytorch 2.0+ (tested with 2.2)
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=$CUDA -c pytorch -c nvidia

# install rdkit
conda install -c conda-forge rdkit -y

# install graph-tool
conda install -c conda-forge graph-tool=2.45 -y

# install ipython and notebook
conda install nb_conda -c conda-forge  -y

# install pytorch-geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+$cuda.html

# install lightning
pip install lightning

# install other packages
pip install yacs tensorboard wandb pyemd pygsp einops

# install molsets
## if use python 3.8, you can try pip 
# pip install molsets
## if use other versions, or you failed installation 
# - please install manually, https://github.com/molecularsets/moses#manually


# recompile orca 
cd pard/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp