#! /bin/bash

# some commands to run other baselines in the paper


# train S-VAE on NNs
python train.py --data-name final_structures6 --save-interval 100 --save-appendix _SVAE --epochs 300 --lr 1e-4 --model SVAE --nz 56 --batch-size 32

# train S-VAE on BNs
python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 50 --save-appendix _SVAE --epochs 100 --lr 1e-4 --model SVAE --nz 56 --batch-size 128

# train GraphRNN on NNs
python train.py --data-name final_structures6 --save-interval 100 --lr 1e-4 --save-appendix _GraphRNN --epochs 300 --model SVAE_GraphRNN --nz 56 --batch-size 32

# train GraphRNN on BNs
python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 50 --save-appendix _GraphRNN --epochs 100 --lr 1e-4 --model SVAE_GraphRNN --nz 56 --batch-size 128

# train GCN on NNs
python train.py --data-name final_structures6 --save-interval 100 --save-appendix _GCN --epochs 300 --lr 1e-4 --model DVAE_GCN --nz 56 --batch-size 32

# train GCN on BNs (need to manually modify DVAE_GCN's default argument to "levels=2" in models.py)
python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 50 --save-appendix _GCN --epochs 100 --lr 1e-4 --model DVAE_GCN --nz 56 --batch-size 128

# train DeepGMG on NNs
python train.py --data-name final_structures6 --save-interval 5 --save-appendix _DeepGMG --epochs 30 --lr 1e-4 --model DVAE_DeepGMG --nz 56 --batch-size 32 --bidirectional

# train DeepGMG on BNs
python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 5 --save-appendix _DeepGMG --epochs 5 --lr 1e-4 --model DVAE_DeepGMG --nz 56 --batch-size 64 --bidirectional

# train D-VAE (fast) on 12-layer NNs
python train.py --data-name final_structures12 --save-interval 50 --lr 1e-4 --save-appendix _DVAE_fast --epochs 800 --batch-size 64 --model DVAE_fast --bidirectional --nz 56
