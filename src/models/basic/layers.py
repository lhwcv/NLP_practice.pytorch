import torch
import torch.nn as nn
import numpy as np


# def _initialize_weights(self):
#     # weight initialization
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0, 0.01)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout):
        super(EmbeddingLayer, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.init_embedding()

    def init_embedding(self,
                       word_emb_array = None,
                       is_static = False):
        #print('embedding size: ', self.emb.weight.data.shape)
        initrange = 0.5
        self.emb.weight.data.uniform_(-initrange, initrange)
        #nn.init.kaiming_normal_(self.emb.weight, mode='fan_out')
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
