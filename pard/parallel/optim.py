import torch.optim as optim
from torch.optim import  Optimizer
import math 

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1,
        min_lr:float = 1e-6,
        min_lr_mode: str ="rescale"
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-5, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        if min_lr > 0.:
            if  min_lr_mode == "clamp":
                lr = max(min_lr/base_lr, lr)
            elif min_lr_mode == "rescale": # "rescale lr"
                lr = (1 - min_lr / base_lr) * lr + min_lr / base_lr

        return lr


    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        last_epoch: int = -1,
        min_lr:float=1e-6,
        min_lr_mode="rescale"
):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        lr = max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        if min_lr > 0.:
            if  min_lr_mode == "clamp":
                lr = max(min_lr/base_lr, lr)
            elif min_lr_mode == "rescale": # "rescale lr"
                lr = (1 - min_lr / base_lr) * lr + min_lr / base_lr
        return  lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
