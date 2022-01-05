#!/bin/bash

. ./path.sh || exit 1;

CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/rt_polarity_cnn_with_glove.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/rt_polarity_cnn_base.yaml

#python main.py --config ./configs/rt_polarity_cnn_base.yaml