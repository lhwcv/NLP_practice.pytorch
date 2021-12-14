import torch
import  torch.nn as nn
from src.models.basic.rnn_cnn_encoders import  RNNEncoder
from src.models.basic.layers import  EmbeddingLayer

def get_rnn_type(rnn_name):
    if rnn_name == 'RNN':
        return nn.RNN
    if rnn_name == 'LSTM':
        return nn.LSTM
    elif rnn_name == 'GRU':
        return  nn.GRU
    else:
        raise RuntimeError(
            "Unexpected rnn_name %s ."
            "rnn_name must be one RNN, LSTM and GRU." %
            rnn_name)

class Text_RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 hidden_n = 128,
                 num_layers = 1,
                 dropout = 0.5,
                 rnn_type = 'LSTM'):
        super(Text_RNN, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,0.0)
        rnn = get_rnn_type(rnn_type)
        self.encoder = RNNEncoder(input_size = embed_dim,
                                  hidden_size = hidden_n,
                                  num_layers = num_layers,
                                  dropout = dropout,
                                  bidirectional = True,
                                  rnn_cls= rnn)
        self.fc = nn.Linear(in_features=self.encoder.get_output_dim(),
                            out_features=hidden_n)
        self.fc_out = nn.Linear(in_features=hidden_n,
                            out_features=num_class)

    def forward(self, text):
        x = self.embedding(text)
        x = self.encoder(x)
        x = torch.tanh(self.fc(x))
        x = self.fc_out(x)
        return x

if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_RNN(vocab_size=100, num_class=2, num_layers=2)
    y = model(x)
    print(y.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))