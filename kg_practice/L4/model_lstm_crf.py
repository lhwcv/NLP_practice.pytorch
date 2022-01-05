import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.basic.layers import  EmbeddingLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from itertools import zip_longest

class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 vocab_size,
                 label_class_size,
                 embed_dim = 300,
                 hidden_dim = 256,
                 num_layers = 1,
                 dropout = 0.1,
                 with_crf = True):
        """

        :param vocab_size:  词表大小
        :param label_class_size:   tag 类别大小
        :param embed_dim:  词嵌入维数
        :param hidden_dim: 隐藏层数
        """
        super(BiLSTM_CRF, self).__init__()

        self.embedding = EmbeddingLayer(vocab_size, embed_dim,0.0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim,
                              num_layers = num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout= dropout if num_layers!=1 else 0.0)

        self.fc = nn.Linear(in_features=2 * hidden_dim,
                            out_features= hidden_dim)
        self.fc_out = nn.Linear(in_features = hidden_dim,
                                out_features = label_class_size)

        #转移矩阵
        self.with_crf = with_crf
        if with_crf:
            self.transition = nn.Parameter(torch.ones(label_class_size, label_class_size) * 1/label_class_size)


    def forward(self, text, lengths):
        emb = self.embedding(text)
        packed = emb
        #packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.bilstm(packed)
        #rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        x = torch.tanh( self.fc(rnn_out) )
        x = self.fc_out(x)

        emission = x
        if not self.with_crf:
            return emission
        #print(emission.shape)
        #[bsize, L, label_class_size]

        # CRF scores: [B, L, label_class_size, label_class_size]
        # 每一个text中的字对应 [label_class_size, label_class_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        _, seq_len, label_class_size = emission.size()
        emission = emission.unsqueeze(2).expand(-1, -1, label_class_size, -1)
        scores = emission + self.transition.unsqueeze(0)

        return scores

    def test(self, text, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(text, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()


            tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token)
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=tag2id['O']))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        #tagids = [list(reversed(i)) for i in tagids]
        #print(tagids)
        return tagids



if __name__ == '__main__':
    x = torch.ones((1, 5), dtype=torch.int64)
    model = BiLSTM_CRF(vocab_size=100, label_class_size=10, with_crf=False)
    lengths = [4]
    y = model(x,lengths)
    print(y.shape)