#!/bin/bash

python bo.py \
  --data-name="asia_200k" \
  --save-appendix="DVAE" \
  --checkpoint=100 \
  --res-dir="BN_results/" \
  --random-as-test \
  --random-baseline \

  #--save-appendix="SVAE" \
  #--save-appendix="GraphRNN" \
  #--save-appendix="GCN" \
  #--save-appendix="DeepGMG" \
