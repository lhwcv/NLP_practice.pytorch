import  os
import torch
import  logging
import argparse
from src.base.default_cfg import get_cfg_defaults
from src.base.common import create_dir_maybe, setup_seed

from kg_practice.L3.model import  Text_CNN


from torchtext.data.utils import(
    get_tokenizer,
    ngrams_iterator,
)

from src.utils.vocab import build_vocab_from_txt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default = os.path.dirname(__file__)+'/./exp/text_cnn_with_glove/',
                        #default=os.path.dirname(__file__) + './exp/text_cnn_with_glove/',
                        type=str,
                        help="")
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "INFO"))

    setup_seed(666)
    cfg = get_cfg_defaults()
    args = get_args()

    if args.model_dir.endswith('\r'):
        args.model_dir = args.model_dir[:-1]

    print('model dir: ', args.model_dir.strip())

    ## check cfg, model, vocab
    model_dir = args.model_dir
    if not os.path.exists(model_dir+'/cfg.yml'):
        raise Exception("can not found cfg.yml in {}".format(model_dir))

    if not os.path.exists(model_dir+'/best.pth'):
        raise Exception("can not found model name: best.pth in {}".format(model_dir))

    if not os.path.exists(model_dir+'/vocab.txt'):
        raise Exception("can not found vocab.txt in {}".format(model_dir))

    cfg.merge_from_file(model_dir+'/cfg.yml')

    device = cfg.SYSTEM.DEVICE
    embed_dim = cfg.MODEL.embed_dim
    num_class = cfg.MODEL.num_classes

    hidden_n = cfg.MODEL.hidden_n
    dropout = cfg.MODEL.dropout
    save_dir = cfg.TRAIN.save_dir


    ## build vocab
    vocab = build_vocab_from_txt(model_dir+'/vocab.txt')
    vocab.set_default_index(vocab['<unk>'])

    tokenizer = get_tokenizer("basic_english")

    ## build model
    print("==> load model from: ", model_dir+'/best.pth')
    model = Text_CNN(vocab_size=len(vocab),
                     embed_dim=embed_dim,
                     num_class=num_class,
                     filter_kerners=[3, 4, 5],
                     hidden_n=hidden_n,
                     dropout=0.0).to(device)
    model.eval()

    model.load_state_dict(torch.load(model_dir+'/best.pth', map_location=device))


    ## demo

    #define pipeline
    def text_pipeline(x):
        return vocab(list(ngrams_iterator(tokenizer(x), ngrams=1)))

    label_maps={
        0: 'positive',
        1: 'negative'
    }

    print("Now you can input a sentence to pred ==> ")
    print("==== for example ===")

    example = 'You just make me so sad and I have to leave you'
    #example = 'a comedy that swings and jostles to the rhythms of life'
    print('[Text]: ', example)

    example = example.lower()
    processed_text = torch.tensor(text_pipeline(example), dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        pred = model(processed_text)
    _,pred = torch.max(pred, -1)
    pred = int(pred)

    print('[Pred]: ', label_maps[pred])

    while True:
        text = input("[Text]: ")
        example = text.lower()
        processed_text = torch.tensor(text_pipeline(example), dtype=torch.int64).unsqueeze(0)
        with torch.no_grad():
            pred = model(processed_text)
        _, pred = torch.max(pred, -1)
        pred = int(pred)
        print('[Pred]: ', label_maps[pred])


