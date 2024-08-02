# Ablation 1: number of steps. No need to train block size prediction 
# python main.py device 5 dataset qm9 task local_denoising diffusion.num_steps 20
# python main.py device 4 dataset qm9 task local_denoising diffusion.num_steps 70
# python main.py device 3 dataset qm9 task local_denoising diffusion.num_steps 100
# python main.py device 3 dataset qm9 task local_denoising diffusion.num_steps 10

# # Grid 
# python main.py device 4 dataset grid task local_denoising
# python main.py device 5 dataset grid-batchseq task local_denoising 
# also need to train two block prediction model. Just use same configuration 



# Ablation 2: max hops, which controls the number of blocks and size of each blocks. 
# larger max hops leads to smaller block size but increased number of blocks. 
# when hops = 0, it reduces to just a single block, hence it's similar to DiGress. 
# For this ablation, let's control the total number of denosing steps. 
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 0
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 1
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 2
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 3
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 4
python main.py device 5 dataset qm9 task local_denoising diffusion.max_hops 5