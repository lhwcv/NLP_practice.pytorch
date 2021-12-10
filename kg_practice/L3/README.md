## L3


- implement TextCNN proposed in https://arxiv.org/pdf/1408.5882.pdf 
- rt-polaritydata movie review data classification task
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
```
cd kg_practice/L3/
sh run.sh

```

in run.sh  we train w/o Glove model
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/rt_polarity_cnn_with_glove.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/rt_polarity_cnn_base.yaml
```
results will save in ./exp/


### details
- model implemented: src/models/text_cnn.py
- rt_polarity with new style in torchtext: src/datasets/rt_polarity.py


### results
model| with_glove| acc
|---|---|:---:
TextCNN| False| 76.92
TextCNN| True| 81.24

### demo predict 

you may fisrt train to get the models.
```
cd kg_practice/L3/
python predict.py

```


```
Now you can input a sentence to pred ==> 
==== for example ===
[Text]:  You just make me so sad and I have to leave you
[Pred]:  negative
[Text]: I love you so much , I am so happy
I love you so much , I am so happy
[Pred]:  positive
[Text]: you make me so sad and unhappy
you make me so sad and unhappy
[Pred]:  negative
```





