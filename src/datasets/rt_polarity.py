from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _download_extract_validate,
    _create_dataset_directory,
    _create_data_from_csv,
)

import re
import os
import logging
import random

logger = logging.getLogger(__name__)

'''
From https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
'''
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def split_data(pos_file,
               neg_file,
               dev_ratio=0.1,
               clean_string=True,
               seed = 666):
    """
    split data into
    """
    datas = []
    # https://blog.csdn.net/weixin_43589681/article/details/85009474
    with open(pos_file, "r",encoding='Windows-1252') as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            datum  = {"y":1,
                      "text": orig_rev}
            datas.append(datum)
    with open(neg_file, "r",encoding='Windows-1252') as f:
        for line in f:
            rev = []
            rev.append(str(line.strip()))
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()

            datum  = {"y":2,
                      "text": orig_rev}
            datas.append(datum)
    random.seed(seed)
    random.shuffle(datas)
    dev_index = int(len(datas) * dev_ratio)
    return  datas[dev_index:], datas[:dev_index]


URL = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

MD5 = '50c1c2c047b4225e148e97aa7708c34e'

_DEV_RATIO = 0.1
_LINES = 10662

_PATH = 'rt-polaritydata.tar.gz'

_EXTRACTED_FILES = {
    'pos': f'{os.sep}'.join(['rt-polaritydata', 'rt-polarity.pos']),
    'neg': f'{os.sep}'.join(['rt-polaritydata', 'rt-polarity.neg']),
}

_EXTRACTED_FILES_MD5 = {
    'pos': "674ca95d7fa6547818f1cb30daa7eafe",
    'neg': "7d312d2c8eacbb57b0074c59b910756a",
}

DATASET_NAME = "RT_Polarity"
NUM_LINES = {
    'train': _LINES - int(_DEV_RATIO * _LINES),
    'dev': int(_DEV_RATIO * _LINES),
}


@_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(('train', 'dev'))
def RT_Polarity(root, split):
    train_cache_name = root + '/train.csv'
    dev_cache_name = root + '/test.csv'

    if not os.path.exists(train_cache_name) or\
        not os.path.exists(dev_cache_name):
    #if True:
        # download first
        paths = []
        for s in ['pos', 'neg']:
            path = _download_extract_validate(root, URL, MD5, os.path.join(root, _PATH),
                                      os.path.join(root, _EXTRACTED_FILES[s]),
                                      _EXTRACTED_FILES_MD5[s], hash_type="md5")
            logging.info("==> downloaded file: {}".format(path))
            paths.append(path)


        # split data and save
        train_datas, dev_datas = split_data(paths[0], paths[1], _DEV_RATIO)
        with open(train_cache_name, 'w') as f:
            for d in train_datas:
                f.write('{} , {}\n'.format(d['y'] , d['text']) )
        logging.info("==> write train data to file {} with {} samples".format(train_cache_name, len(train_datas)))

        with open(dev_cache_name, 'w') as f:
            for d in dev_datas:
                f.write('{} , {}\n'.format(d['y'], d['text']))
        logging.info("==> write dev data to file {} with {} samples".format(train_cache_name, len(dev_datas)))

    filenames_map={
        'train': train_cache_name,
        'dev': dev_cache_name
    }

    logger.info('Creating {} data'.format(split))
    data_iter = _create_data_from_csv(filenames_map[split])
    return _RawTextIterableDataset(DATASET_NAME, NUM_LINES[split], data_iter)

if __name__ =='__main__':
    logging.basicConfig(level=logging.INFO)
    datas = RT_Polarity(root='/home/lhw/data_disk_fast/czcv.haowei/NLU/NLP_practice/src/dataset/classification/data_cache/',
                    split='dev')
    for label, text in datas:
        print(label, text)
