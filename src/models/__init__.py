from src.models.text_cnn import  Text_CNN
from src.models.text_rnn import  Text_RNN
from src.models.text_fasttext import  Text_BowModel
from src.models.text_rcnn import  Text_RCNN
from src.models.text_tcn import  Text_TCN

def build_model(cfg, vocab_size):
    if cfg.MODEL.model_name=='Text_CNN':
        model = Text_CNN(vocab_size = vocab_size,
                        num_class=cfg.MODEL.num_classes,
                        embed_dim = cfg.MODEL.embed_dim,
                        filter_kerners=[3, 4, 5],
                        hidden_n = cfg.MODEL.hidden_n,
                        dropout = cfg.MODEL.dropout)
        return model

    elif cfg.MODEL.model_name=='Text_RNN':
        model = Text_RNN(vocab_size = vocab_size,
                         num_class = cfg.MODEL.num_classes,
                         embed_dim = cfg.MODEL.embed_dim,
                         hidden_n = cfg.MODEL.hidden_n,
                         num_layers = cfg.MODEL.num_layers,
                         dropout = cfg.MODEL.dropout,
                         rnn_type = cfg.MODEL.rnn_type)
        return model

    elif cfg.MODEL.model_name=='Text_BowModel':
        model = Text_BowModel(vocab_size = vocab_size,
                         num_class = cfg.MODEL.num_classes,
                         embed_dim = cfg.MODEL.embed_dim,
                         hidden_n = cfg.MODEL.hidden_n,
                         dropout = cfg.MODEL.dropout)
        return model

    elif cfg.MODEL.model_name=='Text_RCNN':
        model = Text_RCNN(vocab_size = vocab_size,
                         num_class = cfg.MODEL.num_classes,
                         embed_dim = cfg.MODEL.embed_dim,
                         hidden_n = cfg.MODEL.hidden_n,
                         num_layers = cfg.MODEL.num_layers,
                         dropout = cfg.MODEL.dropout,
                         rnn_type = cfg.MODEL.rnn_type)
        return model

    elif cfg.MODEL.model_name=='Text_TCN':
        model = Text_TCN(vocab_size=vocab_size,
                          num_class=cfg.MODEL.num_classes,
                          embed_dim=cfg.MODEL.embed_dim,
                          hidden_n=cfg.MODEL.hidden_n,
                          num_layers=cfg.MODEL.num_layers,
                          dropout=cfg.MODEL.dropout)
        return model
    else:
        raise NotImplementedError("model {} not support now!".format(cfg.MODEL.model_name))
