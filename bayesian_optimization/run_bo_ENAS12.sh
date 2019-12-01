#!/bin/bash

export PYTHONPATH="$(pwd)"

python bo.py \
  --data-name final_structures12 \
  --save-appendix DVAE \
  --checkpoint 800 \
  --res-dir="ENAS12_results/" \
  --BO-rounds 10 \
  --BO-batch-size 50 \
  --random-as-test \
