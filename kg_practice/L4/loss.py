import torch
import torch.nn.functional as F


def _indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets

def cal_crf_loss(crf_scores, targets, label_vocab):
    """计算双向LSTM-CRF模型的损失
    https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = label_vocab['<pad>']
    start_id = label_vocab['<start>']
    end_id = label_vocab['<end>']

    device = crf_scores.device

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(label_vocab)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = _indexed(targets, target_size, start_id)
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                                      t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


