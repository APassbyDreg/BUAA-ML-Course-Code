import torch
from torch import nn, optim
import torch.nn.functional as nnfunc

import os

import Loader as myds
from DNN import *
from LSTM import *


NUM_ITER = 2500
MODEL_PATH = ".\\models\\train_from_raw"
REQUIRE_FLAT = {"dnn_09-24_08-29_0.9688": True, "lstm_09-24_15-59_0.9554": False}
NUM_CLASS = len(REQUIRE_FLAT) * 8
PERCENTAGE_TRAIN = 0.8


def get_ensemble_dataset():
    outputs = []
    _, y = myds.load_all()
    for model_name in REQUIRE_FLAT.keys():
        model = torch.load(os.path.join(MODEL_PATH, model_name))
        x, y = myds.load_all(REQUIRE_FLAT[model_name])
        with torch.no_grad():
            out = model(x)
            outputs.append(out)
    x_new = torch.cat(outputs, dim=1)
    return x_new, y


class Ensemble(nn.Module):
    """
        model that combines multiple pre-trained ones
    """
    def __init__(self):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(NUM_CLASS, 2 * NUM_CLASS)
        self.fc2 = nn.Linear(2 * NUM_CLASS, 8)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = nnfunc.softmax(self.fc2(x), dim=-1)
        return x


if __name__ == "__main__":
    x, y = get_ensemble_dataset()
    x_train = x[:int(x.shape[0]*PERCENTAGE_TRAIN)]
    y_train = y[:int(y.shape[0]*PERCENTAGE_TRAIN)]

    model = Ensemble()
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-2, momentum=0.2)
    lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 250, 500, 1000], gamma=0.5)

    for i in range(NUM_ITER):
        model.zero_grad()
        out = model(x_train)
        loss = loss_func(out, y_train)
        loss.backward()
        optimizer.step()
        lr_decay.step()
        if i % 50 == 49:
            print("iter {:04d}: loss={:.6f}".format(i+1, loss.item()))

    timestamp = datetime.strftime(datetime.now(), "%m-%d_%H-%M")
    file_path = os.path.join(MODEL_PATH, "ensemble_{}".format(timestamp))
    torch.save(model, file_path)

    