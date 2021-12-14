# NLP_practice

We will continuously  update some NLP practice based on different tasks.

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


## Classification

### - Supported models

- [RNN (RNN, LSTM, GRU)](https://arxiv.org/pdf/1408.5882.pdf)
- [TextCNN](https://arxiv.org/pdf/1408.5882.pdf)
- [FastText](https://arxiv.org/pdf/1607.01759.pdf)
- [RCNN](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)
- [TCN](https://arxiv.org/pdf/1803.01271.pdf)

### - Supported datasets

- multi datasets in torchtext.datasets e.g **SougouNews**
- [rt_polarity](src/datasets/rt_polarity.py)

Examples: <br/>

- [Movie Review Classification by TextCNN](./kg_practice/L3/)
  
   actually you can easily modify the dataset name to do experiments on other torchtext's datasets.

- [SougouNews Classification by Multi Models](./kg_practice/L3_extend1)
  
   - TextCNN, RNN, RCNN, FastText, TCN

## TODO MORE TASKS