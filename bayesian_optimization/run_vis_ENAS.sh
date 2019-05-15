#!/bin/bash

python bo.py \
  --data-name="final_structures6" \
  --save-appendix="SVAE56" \
  --checkpoint=300 \
  --vis-2d \
  --res-dir="res56_ENAS_vis2pca/" \
#  --random-as-test \
#  --random-baseline \


  #--save-appendix="DVAE56bi_vid" \
