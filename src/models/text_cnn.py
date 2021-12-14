import torch
import torch.nn as nn
import numpy as np

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

    def init_embedding(self,
                       word_emb_array = None,
                       is_static = False):
        #print('embedding size: ', self.emb.weight.data.shape)
        initrange = 0.5
        self.emb.weight.data.uniform_(-initrange, initrange)

        #
        if word_emb_array is not  None:
            print('==> init embedding with external arr!')
            word_emb_array = torch.from_numpy(word_emb_array)
            assert word_emb_array.shape == self.emb.weight.data.shape
            self.emb.weight.data[:, :] = word_emb_array
        if is_static:
            self.emb.weight.requires_grad = False

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb

'''
https://arxiv.org/pdf/1408.5882.pdf
'''
class Text_CNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 filter_kerners = [3, 4, 5],
                 hidden_n = 128,
                 dropout = 0.5):

        super(Text_CNN, self).__init__()
        self.embedding = WordEmbedding(vocab_size, embed_dim,0.0)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, hidden_n, (f, embed_dim)),
                    nn.ReLU(inplace=True),
                ) for f in filter_kerners
            ])
        self.fc = nn.Linear(in_features  = hidden_n * len(filter_kerners),
                            out_features = num_class)
        self.drop = nn.Dropout(p = dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.init_embedding()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        #nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')

    def forward(self, text):
        #print(text)
        #print('text shape: ',text.shape)

        x = self.embedding(text)
        n, seq_len, embed_dim = x.shape
        x = x.unsqueeze(1) #[n, 1, seq_len, embed_dim]
        pools = []

        for conv in self.convs:
            fea = conv(x)
            # 时序上长度是 seq_len - f +1
            # print(fea.shape)

            ## across word seq
            ksize = fea.shape[-2]
            fea = nn.functional.max_pool2d(fea, kernel_size = (ksize , 1))
            fea = fea.squeeze()

            pools.append(fea)

        x = torch.cat(pools, dim=-1)
        x = self.fc(x)
        x = self.drop(x)
        return  x


if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_CNN(vocab_size=100, num_class=2)
    y = model(x)
    print(y.shape)