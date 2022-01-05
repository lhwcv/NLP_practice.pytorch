import  os
import torch
import  logging
import argparse
from contextlib import redirect_stdout
from src.base.default_cfg import get_cfg_defaults
from src.base.common import create_dir_maybe, setup_seed
from src.datasets import get_dataloader

from torch.nn.utils.rnn import pad_sequence
from kg_practice.L4.data import load_dataset, load_vocab,trans_tags_to_BIO
from kg_practice.L4.model_lstm_crf import BiLSTM_CRF
from kg_practice.L4.loss import cal_crf_loss
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from src.base.logger import TxtLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default = os.path.dirname(__file__)+'./configs/waybill_lstm_crf.yaml',
                        type=str,
                        help="")
    return parser.parse_args()

def eval(model, dloader, label_vocab):
    model.eval()
    id_to_tags = label_vocab.get_itos()
    id_to_tags = trans_tags_to_BIO(id_to_tags)
    pred_tags, gt_tags = [], []

    for texts, targets, lengths in dloader:
        preds = model.test(texts, lengths, label_vocab)

        unpad_labels = [[
            id_to_tags[index]
            for index in targets[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]

        unpad_predictions = [[
            id_to_tags[index]
            for index in preds[sent_index][:lengths[sent_index]]
        ] for sent_index in range(len(lengths))]


        for g, p in zip(unpad_labels, unpad_predictions):
            ## 有时候预测提前输出了 <end>
            ## TODO: 改善这个逻辑不合理， crf decode过程应该保证输出序列长度和输入是一致的
            if len(p) > len(p):
                p = p[:len(g)]
            n = len(g) - len(p)
            if n >0:
                for _ in range(n):
                    p.append(label_vocab['O'])

            gt_tags.append(g[1: -1]) #remove start end
            pred_tags.append(p[1: -1])  # remove start end


    print(classification_report(gt_tags, pred_tags) )

    score = f1_score(gt_tags, pred_tags)
    print("===> f1_score: ", score)
    return score






def main(cfg):
    device = cfg.SYSTEM.DEVICE
    embed_dim = cfg.MODEL.embed_dim

    hidden_n = cfg.MODEL.hidden_n
    dropout = cfg.MODEL.dropout
    save_dir = cfg.TRAIN.save_dir
    data_root = cfg.DATA.data_root
    early_stop_n = cfg.TRAIN.early_stop_n

    create_dir_maybe(save_dir)

    with open(save_dir + '/cfg.yml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    text_vocab = load_vocab(data_root+'/word.dic',specials = ['<start>','<end>','<pad>' ])
    text_vocab.set_default_index(text_vocab["OOV"])

    label_vocab = load_vocab(data_root + '/tag.dic', specials = ['<start>','<end>','<pad>' ])
    label_vocab.set_default_index(label_vocab["O"])

    print('===> len text vocab: ', len(text_vocab))
    print('===> len label vocab: ', len(label_vocab))

    #print(label_vocab.get_stoi())


    ## define pipeline
    def text_pipeline(x): return text_vocab(list(x) )
    def label_pipeline(x): return label_vocab(list(x) )

    ## https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
    # we process the raw text data and add padding to dynamically match the longest sentence in a batch
    def collate_batch(batch):
        label_list, text_list, lengths = [], [], []
        for (_text,_label) in batch:
            #print(_label)
            _text = ['<start>'] + _text + ['<end>']
            _label = ['<start>'] + _label +['<end>']
            lengths.append(len(_text))
            text_list.append(torch.tensor(text_pipeline(_text)) )
            label_list.append(torch.tensor(label_pipeline(_label) ))

        ## batch first
        return pad_sequence(text_list, padding_value=text_vocab['<pad>']).t_(),\
                pad_sequence(label_list, padding_value=label_vocab['<pad>']).t_(),lengths


    train_ds, dev_ds, test_ds = load_dataset(
        datafiles=(os.path.join(data_root, 'train.txt'),
                   os.path.join(data_root, 'dev.txt'),
                   os.path.join(data_root, 'test.txt')))

    train_loader = get_dataloader(train_ds, cfg, collate_fn=collate_batch)
    dev_loader = get_dataloader(dev_ds, cfg, collate_fn=collate_batch)
    test_loader = get_dataloader(test_ds, cfg, collate_fn = collate_batch)

    model = BiLSTM_CRF(vocab_size=len(text_vocab),
                       label_class_size = len(label_vocab),
                       embed_dim=embed_dim,
                       hidden_dim=hidden_n,
                       dropout=dropout).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.TRAIN.learning_rate,
                                 weight_decay=cfg.TRAIN.weight_decay)

    logger = TxtLogger(cfg.TRAIN.save_dir + "/logger.txt")

    step = 0
    best_score = 0.0
    early_n = 1

    for epoch in range(cfg.TRAIN.num_train_epochs):
        model.train()
        for texts, targets, lengths in train_loader:
            crf_scores = model.forward(texts, lengths)
            loss = cal_crf_loss(crf_scores, targets, label_vocab).to(device)

            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            print("[TRAIN] Epoch:%d - Step:%d - Loss: %f" % (epoch, step, loss))
        score = eval(model, dev_loader, label_vocab)
        if best_score < score:
            early_n = 0
            best_score = score
            model_path = os.path.join(save_dir, 'best.pth')
            torch.save(model.state_dict(), model_path)
        else:
            early_n += 1
        logger.write("steps: {} ,mean f1 score : {:.4f} , best f1 score: {:.4f}". \
                          format(step, score, best_score))

        logger.write("==" * 50)

        if early_n > early_stop_n:
            print('dev early stopped!')
            score = eval(model, test_loader, label_vocab)
            logger.write("final best f1 score in test: {:.4f}". \
                         format(score))
            return



if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "INFO"))

    setup_seed(42)
    cfg = get_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())
    cfg.merge_from_file(args.config)

    print(cfg)
    main(cfg)