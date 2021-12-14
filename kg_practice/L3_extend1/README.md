## L3_extend


- SougouNews classification task
- with Glove (6B 300dim) to init embedding for better accuracy

 ### prerequisites

#### Software

- pytorch >= 1.10
- torchtext >= 0.11.0
- sklearn
- tqdm

or

```

cd  NLP_practice/
pip install --no-cache-dir -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/

```

#### Glove
We will download in scripts, but sometimes with network error. <br/>
You can manually download and put in NLP_practice/global_data/ <br/>
From: https://blog.csdn.net/LeYOUNGER/article/details/79343404

---
```
├── global_data
│   ├── glove.6B.100d.txt
│   ├── glove.6B.100d.txt.pt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
│   ├── glove.6B.300d.txt.pt
│   ├── glove.6B.50d.txt
│   └── README.md
```

### train

your network must can access google, otherwise you should download the 
data according to the link in: <br/>

https://github.com/pytorch/text/blob/main/torchtext/datasets/sogounews.py

and put in kg_practice/L3_extend1/data_local/:
```
├── local_data
│   └── SogouNews
│       ├── sogou_news_csv
│       │   ├── classes.txt
│       │   ├── readme.txt
│       │   ├── test.csv
│       │   └── train.csv
│       └── sogou_news_csv.tar.gz
```


Train:

```
cd kg_practice/L3_extend1/
sh run.sh

```

in run.sh  we train multiple models <br/>
```

CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_CNN
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_RNN
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_BowModel
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_RCNN
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/sougou_news_base.yaml --model_name Text_TCN
```
results will save in ./exp/


### results
model| acc
|---|:---:
TextCNN| training..
TextLSTM|  training..
TextFastText| training..
TextRCNN| training..
TextTCN| training..



