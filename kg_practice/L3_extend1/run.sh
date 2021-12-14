#!/bin/bash

. ./path.sh || exit 1;

#CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rt_polarity_base.yaml --model_name Text_CNN
#CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rt_polarity_base.yaml --model_name Text_RNN
#CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rt_polarity_base.yaml --model_name Text_BowModel
#CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rt_polarity_base.yaml --model_name Text_RCNN
#CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/rt_polarity_base.yaml --model_name Text_TCN

CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_CNN
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_RNN
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_BowModel
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_RCNN
CUDA_VISIBLE_DEVICES=1 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_TCN