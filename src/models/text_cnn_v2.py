import torch
import torch.nn as nn
import numpy as np
from src.models.basic.layers import  EmbeddingLayer
from src.models.basic.rnn_cnn_encoders import CNNEncoder

'''
https://arxiv.org/pdf/1408.5882.pdf
'''
class Text_CNN_V2(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 filter_kerners = [3, 4, 5],
                 hidden_n = 128,
                 dropout = 0.5):

        super(Text_CNN_V2, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,0.0)

        self.encoder = CNNEncoder(input_size = embed_dim,
                                  hidden_size = hidden_n,
                                  filter_kerners= filter_kerners,
                                  dropout = dropout)
        self.out_fc = nn.Linear(in_features  = self.encoder.get_output_dim(),
                            out_features = num_class)

        self.init_weights()

    def init_weights(self):
        self.embedding.init_embedding()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text):
        x = self.embedding(text)
        x = self.encoder(x)
        x = torch.tanh(x)
        x = self.out_fc(x)
        return  x


if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_CNN_V2(vocab_size=100, num_class=2)
    y = model(x)
    print(y.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))