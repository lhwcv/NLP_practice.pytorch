import os
from torchtext.vocab import  vocab
from collections import  OrderedDict
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset

def load_vocab(fn, specials = ['<start>','<end>','<pad>' ]):
    d = {}
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            key = line.strip('\n')
            d[key] = 1
    for s in specials:
        d[s] = 1

    return vocab(OrderedDict(d))


def load_dataset(datafiles):
    def read_fn(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)#skip first line
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return to_map_style_dataset(read_fn(datafiles))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [to_map_style_dataset(read_fn(datafile)) for datafile in datafiles]


def trans_tags_to_BIO(tags:list):
    d = {
        "P-B": "B-P",
        "P-I": "I-P",
        "T-B": "B-T",
        "T-I": "I-T",
        "A1-B": "B-A1",
        "A1-I": "I-A1",
        "A2-B": "B-A2",
        "A2-I": "I-A2",
        "A3-B": "B-A3",
        "A3-I": "I-A3",
        "A4-B": "B-A4",
        "A4-I": "I-A4",
        "O": "O"
    }
    return  [d.get(t, t) for t in tags]

if __name__ == '__main__':
    fn = './local_data/waybill/test.txt'
    ds = load_dataset(fn)
    for item in ds:
        print(item)
        break

    label_vocab = load_vocab('./local_data/waybill//tag.dic')
    label_vocab.set_default_index(label_vocab["O"])

    print(label_vocab.get_stoi())

