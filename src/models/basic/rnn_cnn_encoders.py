import  torch
import torch.nn  as nn

class RNNEncoder(nn.Module):
    r"""
        Args:
            input_size (int):
                The number of expected features in the input (the last dimension).
            hidden_size (int):
                The number of features in the hidden state.
            num_layers (int, optional):
                Number of recurrent layers.
                Defaults to 1.
            bidirectional (bool, optional):
                Defaults to False.
            dropout (float, optional):
                Defaults to 0.0.
            pooling_type (str, optional):
                If `pooling_type` is None, then the RNNEncoder will return the hidden state of
                the last time step at last layer as a single vector.
                If pooling_type is not None, it must be one of "sum", "max" and "mean".
                Then it will be pooled on the LSTM output (the hidden state of every time
                step at last layer) to create a single vector.
                Defaults to `None`
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers = 1,
                 dropout = 0.0,
                 bidirectional = True,
                 pooling_type = None,
                 rnn_cls = nn.LSTM):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.pooling_type = pooling_type
        self.rnn_layers = rnn_cls(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout,
                            bidirectional = bidirectional)
    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        if self.bidirectional:
            return  2 * self.hidden_size
        return self.hidden_size

    def forward(self, x):
        """

        :param x:  [batch_size, seq_len, input_size]
        :return:  [batch_size, hidden_size]
        """
        out, _ = self.rnn_layers(x)

        if self.pooling_type is not  None:
            if self.pooling_type == 'mean':
                out = torch.mean(out, dim = 1)
                return  out
            elif self.pooling_type == 'max':
                out,_ = torch.max(out, dim=1)
                return out
            raise RuntimeError(
                "Unexpected pooling type %s ."
                "Pooling type must be one of max and mean." %
                self.pooling_type)

        return out[:, -1, :]

'''
https://arxiv.org/pdf/1408.5882.pdf
'''
class CNNEncoder(nn.Module):
    def __init__(self,
                 input_size = 300,
                 hidden_size=128,
                 filter_kerners = [3, 4, 5],
                 dropout = 0.0):
        super(CNNEncoder, self).__init__()
        self.filter_kerners  = filter_kerners
        self.hidden_size = hidden_size
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, hidden_size, (f, input_size)),
                    #nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                ) for f in filter_kerners
            ])
        self.drop = nn.Dropout(p = dropout)

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * len(self.filter_kerners)

    def forward(self, x):
        """
        :param x:  [batch_size, seq_len, input_size]
        :return:  [batch_size, hidden_size]
        """
        x = x.unsqueeze(1) #[n, 1, seq_len, embed_dim]
        pools = []

        for conv in self.convs:
            fea = conv(x)
            ## across word seq
            ksize = fea.shape[-2]
            fea = nn.functional.max_pool2d(fea, kernel_size = (ksize , 1))
            fea = fea.squeeze()

            pools.append(fea)

        x = torch.cat(pools, dim=-1)
        x = self.drop(x)
        return  x


if __name__ == '__main__':
    x = torch.ones((4, 10, 64), dtype=torch.float32)
    #model = RNNEncoder(64, 128, rnn_cls= nn.GRU)
    model = CNNEncoder(64, 128)
    y = model(x)
    print(y.shape)

