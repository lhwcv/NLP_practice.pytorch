import importlib
from  torch.utils.data import  Dataset,DataLoader
from torchtext.datasets import DATASETS
from .rt_polarity import RT_Polarity

DATASETS['RT_Polarity'] = RT_Polarity

URLS = {}
NUM_LINES = {}
MD5 = {}

for dataset in DATASETS:
    try:
        dataset_module_path = "torchtext.datasets." + dataset.lower()
        dataset_module = importlib.import_module(dataset_module_path)
    except:
        dataset_module_path = "src.datasets." + dataset.lower()
        dataset_module = importlib.import_module(dataset_module_path)

    URLS[dataset] = dataset_module.URL
    NUM_LINES[dataset] = dataset_module.NUM_LINES
    MD5[dataset] = dataset_module.MD5

__all__ = sorted(list(map(str, DATASETS.keys())))


def get_dataset(dataset_name,data_dir, split = 'train'):
    if dataset_name not in DATASETS.keys():
        raise  NotImplementedError('Dataset Type not supported!')
    return  DATASETS[dataset_name](root = data_dir, split=split)

def get_dataloader(dataset_ins, cfg, collate_fn=None):
    dataloader = DataLoader(
        dataset_ins,
        batch_size = cfg.TRAIN.batch_size,
        shuffle = False,
        num_workers = cfg.SYSTEM.NUM_WORKERS,
        drop_last=False,
        collate_fn = collate_fn)
    return  dataloader