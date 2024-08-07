# Ablation 1: number of steps. No need to train block size prediction 
# python main.py device 5 dataset qm9 task local_denoising diffusion.num_steps 20
# python main.py device 4 dataset qm9 task local_denoising diffusion.num_steps 70
# python main.py device 3 dataset qm9 task local_denoising diffusion.num_steps 100
# python main.py device 3 dataset qm9 task local_denoising diffusion.num_steps 10

# # Grid 
# python main.py device 4 dataset grid task local_denoising
# python main.py device 5 dataset grid-batchseq task local_denoising 

# Ablation 2: max hops, which controls the number of blocks and size of each blocks. 
# larger max hops leads to smaller block size but increased number of blocks. 
# when hops = 0, it reduces to just a single block, hence it's similar to DiGress. 
# For this ablation, let's control the total number of denosing steps to be 140. 
## Part 1: block_prediction
# python main.py device 0 dataset qm9 task block_prediction diffusion.max_hops 0 
python main.py device 3 dataset qm9 task block_prediction diffusion.max_hops 1 &
python main.py device 4 dataset qm9 task block_prediction diffusion.max_hops 2 &
# python main.py device 3 dataset qm9 task block_prediction diffusion.max_hops 3 
python main.py device 5 dataset qm9 task block_prediction diffusion.max_hops 4 &
# python main.py device 5 dataset qm9 task block_prediction diffusion.max_hops 5 
## Part 2: diffusion block_prediction
# python main.py device 0 dataset qm9 task local_denoising diffusion.max_hops 0 diffusion.num_steps 140 # Number of blocks: 1,  max block size: 30, max block degree: 5 | Average number of blocks in training set: 1.0
# python main.py device 1 dataset qm9 task local_denoising diffusion.max_hops 1 diffusion.num_steps 32  # Number of blocks: 8,  max block size: 21, max block degree: 4 | Average number of blocks in training set: 4.316677421545986
# python main.py device 2 dataset qm9 task local_denoising diffusion.max_hops 2 diffusion.num_steps 25  # Number of blocks: 10, max block size: 21, max block degree: 4 | Average number of blocks in training set: 5.656101412851514
# python main.py device 3 dataset qm9 task local_denoising diffusion.max_hops 3 diffusion.num_steps 20  # Number of blocks: 13, max block size: 19, max block degree: 4 | Average number of blocks in training set: 7.200592650455101
# python main.py device 4 dataset qm9 task local_denoising diffusion.max_hops 4 diffusion.num_steps 18  # Number of blocks: 15, max block size: 19, max block degree: 4 | Average number of blocks in training set: 7.752029275913599
# python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 5 diffusion.num_steps 18  # Number of blocks: 15, max block size: 19, max block degree: 4 | Average number of blocks in training set: 7.838498505637821

python main.py device 4 dataset qm9 task local_denoising diffusion.max_hops 3 diffusion.num_steps 20 model.transformer_only True model.edge_hidden 80
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 3 diffusion.num_steps 20 model.hidden_size 128 True model.edge_hidden 0