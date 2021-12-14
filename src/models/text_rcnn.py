import torch
import  torch.nn as nn
from src.models.basic.layers import  EmbeddingLayer
from src.models.text_rnn import get_rnn_type

"""
Recurrent Convolutional Neural Networks for Text Classification
https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552

"""
class Text_RCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 hidden_n = 128,
                 num_layers = 1,
                 dropout = 0.0,
                 rnn_type = 'LSTM'):
        super(Text_RCNN, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim,0.0)
        rnn_cls = get_rnn_type(rnn_type)
        self.rnn = rnn_cls(input_size = embed_dim,
                                  hidden_size = hidden_n,
                                  num_layers = num_layers,
                                  dropout = dropout,
                                  bidirectional = True)
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(2 * hidden_n + embed_dim, hidden_n, 1)
        self.fc = nn.Linear(in_features= hidden_n,
                            out_features=num_class)

    def forward(self, text):
        embed = self.embedding(text)
        x,_ = self.rnn(embed)
        fea = torch.cat((embed, x), 2) #(b, seq_len, 2 * hidden_size + embed_dim)

        fea = fea.permute(0, 2, 1)
        fea = torch.relu(self.conv(fea) ) #(b, hidden_size, seq_len)
        fea = self.drop(fea)

        ksize = fea.shape[-1]
        fea = nn.functional.max_pool2d(fea, kernel_size=(1, ksize))
        fea = fea.squeeze()
        out = self.fc(fea)
        return out

if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_RCNN(vocab_size=100, num_class=2, num_layers=2)
    y = model(x)
    print(y.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))