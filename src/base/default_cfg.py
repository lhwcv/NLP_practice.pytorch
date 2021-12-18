from yacs.config import CfgNode as CN
__all_ = ['get_cfg_defaults']

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 2
_C.SYSTEM.DEVICE = 'cpu'
_C.SYSTEM.global_data_root = '../../data/'


_C.DATA = CN()
_C.DATA.dataset_name = 'ExpW'
_C.DATA.data_root = './data/'
_C.DATA.min_freq = 1

_C.TRAIN = CN()
_C.TRAIN.batch_size = 4
_C.TRAIN.save_dir = './exp/'
_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_train_epochs = 30
_C.TRAIN.warmup_steps = 100
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.milestones = [20, 50]
_C.TRAIN.milestones_in_epo = True
_C.TRAIN.lr_decay_gamma = 0.2
_C.TRAIN.weight_decay = 0.0
_C.TRAIN.log_steps = 200
_C.TRAIN.adam_epsilon = 1e-6
_C.TRAIN.early_stop_n = 4
_C.TRAIN.load_from=''
_C.TRAIN.milestones_in_epo=False

_C.TEST = CN()
_C.TEST.batch_size = 256
_C.TEST.model_load_path = ''
_C.TEST.save_dir = ''

_C.MODEL = CN()
_C.MODEL.model_name = 'Text_CNN'
_C.MODEL.num_classes = 2
_C.MODEL.embed_dim = 300
_C.MODEL.max_seq_len = 50
_C.MODEL.init_embed_with_glove = True
_C.MODEL.hidden_n = 128
_C.MODEL.dropout = 0.5
_C.MODEL.num_layers = 1
_C.MODEL.rnn_type = "LSTM"

def get_cfg_defaults(merge_from = None):
  cfg =  _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg
