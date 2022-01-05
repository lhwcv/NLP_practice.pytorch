import  os
import torch
import  logging
import argparse
from contextlib import redirect_stdout
from src.base.default_cfg import get_cfg_defaults
from src.base.common import create_dir_maybe, setup_seed
from src.datasets import get_dataset,get_dataloader
from src.utils.word_vectors import get_word_vectors_array_by_glove

from kg_practice.L3_extend1.model import  build_model


from torchtext.data.utils import(
    get_tokenizer,
    ngrams_iterator,
)

from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.functional import to_map_style_dataset

from src.optim.lr_scheduler import WarmupMultiStepLR
from src.base.logger import TxtLogger
from tasks.classification.trainval import Text_Classification_Learner


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default = os.path.dirname(__file__)+'./configs/rt_polarity_cnn_base.yaml',
                        type=str,
                        help="")
    parser.add_argument("--model_name",
                        default='Text_CNN',
                        type=str,
                        help="")
    return parser.parse_args()

def main(cfg):

    ngrams = 2 if cfg.MODEL.model_name=='Text_BowModel' else 1
    device = cfg.SYSTEM.DEVICE
    embed_dim = cfg.MODEL.embed_dim
    save_dir = cfg.TRAIN.save_dir
    max_seq_len = 2 * cfg.MODEL.max_seq_len if cfg.MODEL.model_name=='Text_BowModel' else cfg.MODEL.max_seq_len
    min_freq = cfg.DATA.min_freq

    create_dir_maybe(save_dir)

    with open(save_dir + '/cfg.yml', 'w') as f:
        with redirect_stdout(f): print(cfg.dump())

    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter, ngrams):
        for _, text in data_iter:
            yield ngrams_iterator(tokenizer(text), ngrams)

    train_iter = get_dataset(cfg.DATA.dataset_name, cfg.DATA.data_root, 'train')

    vocab = build_vocab_from_iterator(yield_tokens(train_iter, ngrams),
                                      min_freq = min_freq,
                                      specials = ["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    with open(save_dir+'/vocab.txt','w') as f:
        for s in vocab.get_itos():
            f.write(s+'\n')

    ## define pipeline
    def text_pipeline(x): return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))
    def label_pipeline(x): return int(x) - 1

    # def collate_batch(batch):
    #     label_list = []
    #     text_tensor = torch.ones( (len(batch), max_seq_len), dtype=torch.int64)
    #     text_tensor[:,:] = vocab.get_default_index()
    #     for _inx, (_label, _text) in enumerate(batch):
    #         label_list.append(label_pipeline(_label))
    #         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
    #         if processed_text.size(0) > max_seq_len:
    #             text_tensor[_inx] = processed_text[:max_seq_len]
    #         else:
    #             text_tensor[_inx][:processed_text.size(0)] = processed_text
    #
    #     label_list = torch.tensor(label_list, dtype=torch.int64)
    #     return label_list, text_tensor

    ## https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb
    # we process the raw text data and add padding to dynamically match the longest sentence in a batch
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text))
            text_list.append(processed_text)
        #print(text_list)
        ## batch first
        label =  torch.tensor(label_list)
        text  =  pad_sequence(text_list, padding_value=vocab.get_default_index()).t_()
        if text.size(1) > max_seq_len:
            text = text[:, :max_seq_len]
        return label, text


    train_iter = get_dataset(cfg.DATA.dataset_name, cfg.DATA.data_root, 'train')
    dev_iter = get_dataset(cfg.DATA.dataset_name,
                           cfg.DATA.data_root,
                           'dev' if cfg.DATA.dataset_name=='RT_Polarity' else 'test')

    train_dataset = to_map_style_dataset(train_iter)
    dev_dataset = to_map_style_dataset(dev_iter)

    train_loader = get_dataloader(train_dataset, cfg, collate_fn = collate_batch, shuffle=True)
    dev_loader   = get_dataloader(dev_dataset, cfg, collate_fn = collate_batch)

    ## model and word vectors
    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    model = model.to(device)

    if cfg.MODEL.init_embed_with_glove:
        print('==> init embedding with glove..')
        arr = get_word_vectors_array_by_glove(vocab,
                                              cache_dir = cfg.SYSTEM.global_data_root,
                                              dim = embed_dim)
        print('==> embeding shape: ', arr.shape)
        model.embedding.init_embedding(arr, is_static=False)

    if cfg.TRAIN.milestones_in_epo:
        ns = len(train_loader)
        milestones = []
        for m in cfg.TRAIN.milestones:
            milestones.append(m * ns)
        cfg.TRAIN.milestones = milestones

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.TRAIN.learning_rate,
                                 weight_decay=cfg.TRAIN.weight_decay)
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=cfg.TRAIN.milestones,
        gamma=cfg.TRAIN.lr_decay_gamma,
        warmup_iters=cfg.TRAIN.warmup_steps,
    )
    logger = TxtLogger(cfg.TRAIN.save_dir + "/logger.txt")

    learner = Text_Classification_Learner(
        cfg,
        model=model,
        loss_fn = torch.nn.functional.cross_entropy,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        logger=logger,
        save_dir=cfg.TRAIN.save_dir,
        log_steps=cfg.TRAIN.log_steps,
        device = device,
        gradient_accum_steps = 1,
        max_grad_norm = 1.0,
        early_stop_n=cfg.TRAIN.early_stop_n)

    learner.train(train_loader, dev_loader, epoches=cfg.TRAIN.num_train_epochs)



if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "INFO"))

    setup_seed(666)
    cfg = get_cfg_defaults()
    args = get_args()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())
    cfg.merge_from_file(args.config)

    cfg.MODEL.model_name = args.model_name
    cfg.TRAIN.save_dir = cfg.TRAIN.save_dir + args.model_name
    print(cfg)
    main(cfg)