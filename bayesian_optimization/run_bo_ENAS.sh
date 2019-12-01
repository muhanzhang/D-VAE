#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo.py \
  --data-name final_structures6 \
  --save-appendix DVAE \
  --checkpoint 300 \
  --res-dir="ENAS_results/" \
  --BO-rounds 10 \
  --BO-batch-size 50 \
  --random-as-test \
  --random-baseline \

  #--save-appendix SVAE \
  #--save-appendix GraphRNN \
  #--save-appendix GCN \
  #--save-appendix DeepGMG \
  #--save-appendix DVAE_fast \

