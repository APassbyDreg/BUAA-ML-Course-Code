import torch
import numpy as np

import os
import json


DATA_PATH = ".\\data"
TRAINING_SET_NAME = "#train"
TEST_SET_NAME = "#test"


def load_dataset(ds_name, require_flat=False, use_extended=False):
    """
        load dataset by ds_name, return 2d tensors

        ds_name only take "train" or "test" as input
    """
    if ds_name == 'train':
        ds_path = os.path.join(DATA_PATH, TRAINING_SET_NAME)
    elif ds_name == 'test':
        ds_path = os.path.join(DATA_PATH, TEST_SET_NAME)
    else:
        err_msg = "invalid ds_name: '{}'\nrequire either 'train' or 'test'".format(ds_name)
        raise ValueError(err_msg)
    
    if use_extended:
        ds_path += "_expanded"

    # load ds
    fp = open(ds_path + ".json", "r")
    ds_raw = np.array(json.load(fp), dtype=object).T

    # convert to tensors and return    
    x = torch.tensor(ds_raw[0].tolist(), requires_grad=True, dtype=torch.float)
    y = torch.tensor(ds_raw[1].tolist(), requires_grad=False, dtype=torch.float)
    if require_flat:
        x = x.view(x.shape[0], -1)
    return x, y


def load_all(require_flat=False, use_extended=False):
    x_train, y_train = load_dataset("train", require_flat, use_extended)
    x_test, y_test = load_dataset("test", require_flat, use_extended)
    x_all = torch.cat((x_train, x_test))
    y_all = torch.cat((y_train, y_test))
    return x_all, y_all
    

if __name__ == "__main__":
    load_dataset('test')