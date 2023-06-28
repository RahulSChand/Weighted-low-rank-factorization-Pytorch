# Weighted-low-rank-factorization-Pytorch
PyTorch implementation of the paper Language model compression with weighted low-rank factorization (https://arxiv.org/abs/2207.00112 ICLR 2022)

Open source implementation of FWSVD(Factorized weighted Singular Value decompisition) training method introduced in https://arxiv.org/abs/2207.00112 using pytorch, accelerate & HuggingFace 

Run `train_glue_fwsvd.sh` to run a pre-trained BERT 12 layer model on GLUE tasks with FWSVD. Use `TASK_NAME` flag to change the task
