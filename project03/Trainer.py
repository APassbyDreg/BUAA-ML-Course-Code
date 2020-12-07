import torch
from torch import nn, optim

from DiffModel import *

import json
import time
from datetime import datetime


DS_EXTEND = 25
BATCH_SIZE = 25000
BATCH_NUM = 5000 * 2 * DS_EXTEND / BATCH_SIZE
N_ITER = 300
PROGRESS_LEN = 40
ACCURACY_THRES = 90


def progress(n_batch):
    """
        return a progress bar based on curr #batch
    """
    p = n_batch / BATCH_NUM
    num_eq = round(p * PROGRESS_LEN)
    num_space = PROGRESS_LEN - num_eq
    return "="*num_eq + ">" + " "*num_space
    

def get_eta(n_epoch, n_batch, avg_time):
    batch_done = n_epoch * BATCH_NUM + n_batch
    batch_total = N_ITER * BATCH_NUM
    remain_ms = int((batch_total - batch_done) * avg_time * 1000)
    remain_s = round((remain_ms % 60000) / 1000)
    remain_m = (remain_ms // 60000) % 60
    remain_h = remain_ms // 3600000
    eta = "{:02d}s".format(remain_s)
    if remain_m > 0:
        eta = "{:02d}m_".format(remain_m) + eta
    if remain_h > 0:
        eta = "{:02d}h_".format(remain_h) + eta
    return eta


def train(ds):
    # for parallel computing
    torch.multiprocessing.freeze_support()


    # load dataset and prepare loader
    ds_train = DiffModelDataset(ds["train"], extend=DS_EXTEND)
    ds_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    BATCH_NUM = len(ds_train) / BATCH_SIZE


    # define model, optimizers, etc
    model = DiffModel()
    model.to(dev)
    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.8)
    # lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
    # lr_decay = optim.lr_scheduler.MultiStepLR(optimizer,[20, 50, 90, 150, 250], 0.4)
    lr_decay = optim.lr_scheduler.StepLR(optimizer, 20, 0.4)


    # train
    best_loss = float("inf")
    best_model = DiffModel()
    best_epoch = None
    for n_epoch in range(N_ITER):
        n_batch = 0
        avg_time = 0
        start = time.monotonic()
        epoch_loss = 0
        for x1, x2, y in ds_loader:
            x1.requires_grad = True
            x2.requires_grad = True
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            y = y.to(dev)
            optimizer.zero_grad()
            out = model(x1, x2)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            end = time.monotonic()
            # show output
            if avg_time is None:
                avg_time = end - start
            else:
                avg_time = (avg_time * n_batch + (end - start)) / (n_batch + 1)
            epoch_loss = (epoch_loss * n_batch + loss.item()) / (n_batch + 1)
            n_batch += 1
            eta = get_eta(n_epoch, n_batch, avg_time)
            print("epoch {:-4d} | {} | batch {:-3d}: loss={:.6f} | average time used: {:.4f}s, eta: {}".format(n_epoch+1, progress(n_batch), n_batch, loss.item(), avg_time, eta) + " "*10, end="\r")
            start = time.monotonic()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = n_epoch + 1
            best_model.load_state_dict(model.state_dict())
        print("")
        lr_decay.step()


    # save model and brief
    print("best model is from epoch {} with loss={}".format(best_epoch, best_loss))
    timestamp = datetime.strftime(datetime.now(), "%m-%d_%H-%M")
    file_path = "models/results/diff_model_{}_{}.pkl".format(best_model.version, timestamp)
    torch.save(best_model, file_path)

    accuracy, testlogs = test(ds, best_model)
    with open("models/results/diff_model_{}_{}_info.txt".format(best_model.version, timestamp), "w") as f:
        f.writelines(testlogs)

    return best_model


def test(ds, model):
    model.to(dev)
    model.eval()
    logs = [str(model)+"\n"]
    success = 0
    for idx in range(len(ds['valid'])):
        target = ds['valid'][idx][0]
        pred, confidence = predict(ds['train'], target, model)
        log = "class {:02d}: predicted as {:02d} with confidence {:.4f}".format(idx, pred, confidence)
        print(log)
        logs.append(log+"\n")
        if pred == idx:
            success += 1
    accuracy = success / len(ds['valid'])
    print("accuracy: {:.4f}".format(accuracy))
    logs.append("accuracy: {:.4f}\n".format(accuracy))
    return accuracy, logs


if __name__ == "__main__":
    ds_path = r".\myds\post_res50_expanded.json"
    ds = json.load(open(ds_path, "r"))

    model = train(ds)
    # model = torch.load(r"models\results\diff_model_v1_11-10_08-13.pkl")

    # model.eval()
    accuracy, logs = test(ds, model)
    # with open(r"models\results\diff_model_v1_11-10_08-13_info.txt", "w") as f:
    #     f.writelines(logs)