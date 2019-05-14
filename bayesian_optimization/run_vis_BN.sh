#!/bin/bash

python bo.py \
  --data-name="asia_200k" \
  --save-appendix="SVAE" \
  --checkpoint=100 \
  --vis-2d \
  --res-dir="2dvis_results/" \

  #--save-appendix="DVAE_BN56" \
