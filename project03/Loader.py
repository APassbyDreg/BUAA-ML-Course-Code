import json
from os import path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import os
import torchvision


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


ds = ".\\BMP600"
test_ds = ".\\BMP600_test"
name_format = "{:03d}_{:d}.bmp"
n_id = 100
n_sample = 6
preprocess = transforms.Compose([
    # transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def sort_folder():
    for idx in range(n_id):
        original_path = os.path.join(ds, "{:03d}".format(idx+1))
        new_path = os.path.join(test_ds, "{:03d}".format(idx+1))
        os.system("mkdir " + new_path)
        for sample in range(5, n_sample):
            name = name_format.format(idx+1, sample+1)
            os.system("move {} {}".format(os.path.join(original_path, name), os.path.join(new_path, name)))


def get_loader(batch_size=50, use_test=False):
    if use_test:
        ds_path = test_ds
    else:
        ds_path = ds
    dataset = torchvision.datasets.ImageFolder(ds_path, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def ds_after_resnet(resnet, data_loader):
    x_all = []
    y_all = []
    resnet.to(device)
    for data, labels in data_loader:
        bs = labels.size()[0]
        labels = labels.view(bs, 1)
        y = torch.zeros(bs, n_id)
        y = y.scatter(1, labels, 1)
        data = data.to(device)
        x = resnet(data)
        x_all.extend(x.tolist())
        y_all.extend(y.tolist())
        print("progress: {}/6000".format(len(x_all)))
    return x_all, y_all


if __name__ == "__main__":
    # x,y = load()
    # print(x)
    # print(y)
    # sort_folder()
    bs = 200
    loader_train = get_loader(batch_size=bs)
    loader_valid = get_loader(use_test=True, batch_size=bs)
    res50 = torch.load(r"models\pretrained\resnet50_pretrained")
    res50.to(device)
    print("resources loaded")

    train = [[] for i in range(100)]
    valid = [[] for i in range(100)]

    with torch.no_grad():
        size = 0
        for x, y in loader_train:
            x = x.to(device)
            post = res50(x)
            for i in range(len(y)):
                train[y[i]].append(post[i].tolist())
                size += 1
            print("loaded {} / {} from train".format(size, len(loader_train) * bs))
        size = 0
        for x, y in loader_valid:
            x = x.to(device)
            post = res50(x)
            for i in range(len(y)):
                valid[y[i]].append(post[i].tolist())
                size += 1
            print("loaded {} / {} from valid".format(size, len(loader_valid) * bs))
    print("data transformed")

    a = []
    f = open("newest_post_res50.json", "w")
    json.dump({"train": train, "valid": valid, "all": a}, f)
    
