# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/gpt2-124M-differential.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-differential'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 30 gradaccum = 491520 
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 30
dropout = 0.1
learning_rate = 1e-3

# this makes total number of tokens be 49B
warmup_iters = 1000
max_iters = 100000
lr_decay_iters = 40000  # 19B tokens, or about 2 epochs
min_lr = 1e-6

# eval stuff
eval_interval = 100
eval_iters = 25
log_interval = 10

bias = False

# weight decay
weight_decay = 1e-1
