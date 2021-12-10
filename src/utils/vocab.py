from torchtext.vocab import  vocab
from collections import  OrderedDict

def build_vocab_from_txt(filepath):
    d = {}
    with open(filepath, encoding='utf-8') as f:
        for text in f:
            d[text.strip()] = 1
    return  vocab(OrderedDict(d))