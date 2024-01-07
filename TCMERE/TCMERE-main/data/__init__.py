from constants import *
from data.base import DataInstance
from data.helpers import tokenize
from data.biorelex import load_biorelex_dataset

def load_data(dataset, split_nb, tokenizer):
    assert (dataset in DATASETS)
    base_path = 'resources/{}'.format(dataset)
    if dataset == BIORELEX:
        return load_biorelex_dataset(base_path, tokenizer)
