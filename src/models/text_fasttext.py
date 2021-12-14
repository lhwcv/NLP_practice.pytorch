import torch
import  torch.nn as nn
from src.models.basic.layers import  EmbeddingLayer
"""
Bag of Tricks for Efficient Text Classification
https://arxiv.org/pdf/1607.01759.pdf
"""

class Text_BowModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 hidden_n = 128,
                 dropout = 0.0):
        super(Text_BowModel, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,0.0)
        self.fc1 = nn.Linear(in_features = embed_dim,
                            out_features = hidden_n)
        self.drop = nn.Dropout(dropout)
        self.fc_out = nn.Linear(in_features=hidden_n,
                             out_features=num_class)

    def forward(self, text):
        x = self.embedding(text)
        ksize = x.shape[-2]
        fea = nn.functional.avg_pool2d(x, kernel_size=(ksize, 1))
        fea = fea.squeeze()
        fea = torch.relu(fea)
        out = self.fc1(fea)
        out = torch.relu(out)
        out = self.drop(out)
        out = self.fc_out(out)
        return out

if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_BowModel(vocab_size=100, num_class=2)
    y = model(x)
    print(y.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))