"""
https://arxiv.org/pdf/1803.01271.pdf

refer from: https://github.com/locuslab/TCN
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.models.basic.layers import  EmbeddingLayer

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Text_TCN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_class,
                 embed_dim = 100,
                 hidden_n = 128,
                 num_layers = 2,
                 kernel_size = 2,
                 dropout =0.3):
        super(Text_TCN, self).__init__()

        num_channels = [hidden_n ] * num_layers
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, 0.0)
        self.tcn = TemporalConvNet(embed_dim, num_channels, kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], num_class)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        self.embedding.init_embedding()
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, text):
        emb = self.embedding(text)
        x = self.tcn(emb.permute(0, 2, 1))
        #print(x.shape)
        x = x.mean(dim=-1)
        x = torch.relu(x)
        #x = self.drop(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    x = torch.ones((4, 10), dtype=torch.int64)
    model = Text_TCN(vocab_size=100, num_class=2, num_layers=2)
    y = model(x)
    print(y.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print('Total params: %.2fM' % (params / 1000000.0))
    print('Total flops: %.2fM' % (flops / 1000000.0))