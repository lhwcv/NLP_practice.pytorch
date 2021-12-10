from torchtext.vocab import GloVe,Vocab,vocab
from collections import OrderedDict
import numpy as np

def get_word_vectors_array_by_glove(vocab1: Vocab,
                                    cache_dir = '/home/lhw/nlp_data/',
                                    name = '6B',
                                    dim = 100):
    glove = GloVe(name=name, cache=cache_dir, dim=dim)
    tokens = vocab1.get_itos()
    vecs = glove.get_vecs_by_tokens(tokens, lower_case_backup=True)
    arr = np.array(vecs, np.float32)
    return arr

if __name__ == '__main__':
    tokens = ['hello', 'world']
    d = OrderedDict([(token, 1) for token in tokens])
    v2 = vocab(d)
    unk_token = '<unk>'
    v2.insert_token(unk_token, 0)
    print(v2.get_itos())

    print('vocab size: ', len(v2))
    arr = get_word_vectors_array_by_glove(v2)
    print(arr.shape)