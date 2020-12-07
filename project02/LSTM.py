import torch
from torch import nn, optim
import torch.nn.functional as nnfunc

import os
from datetime import datetime

import Loader as myds


device = torch.device("cuda:0")

N_ITER = 300
MODEL_PATH = ".\\models\\train_from_extended"


class LSTM(nn.Module):
    """
        simple LSTM model
    """

    def __init__(self):
        super(LSTM, self).__init__()
        self.preprocess = nn.Linear(6, 6)
        self.lstm = nn.LSTM(6, 8)
        self.fc1 = nn.Linear(1600, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 8)

    def forward(self, x):
        x = nnfunc.relu(self.preprocess(x))
        x, _ = self.lstm(x)
        x = nnfunc.relu(self.fc1(x.view(x.shape[0], -1)))
        x = nnfunc.relu(self.fc2(x))
        x = nnfunc.softmax(self.fc3(x), dim=-1)
        return x


if __name__ == "__main__":
    # prepare
    lstm = LSTM()
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(lstm.parameters(), lr=1e-3, momentum=0.6)
    lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.6)

    x, y = myds.load_dataset("train", require_flat=False, use_extended=True)
    test_x, test_y = myds.load_dataset("test", require_flat=False, use_extended=True)

    # train
    for i in range(N_ITER):
        lstm.zero_grad()
        out = lstm(x)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        lr_decay.step()
        # show
        if True:
            with torch.no_grad():
                predict = lstm(test_x)
                test_loss = loss_func(predict ,test_y)
                print("iter {:03d}: loss={:.6f}, test_set_loss={:.6f}".format(i+1, loss.item(), test_loss.item()))

    # check accuracy
    with torch.no_grad():
        predict = lstm(test_x)
        correct = 0
        for i in range(test_x.size()[0]):
            if torch.argmax(predict[i], dim=-1) == torch.argmax(test_y[i], dim=-1):
                correct += 1
        accuracy = correct / test_x.size()[0]
        print("accuracy is {:.4f} ({}/{})".format(accuracy, correct, test_x.size()[0]))

    # save model
    timestamp = datetime.strftime(datetime.now(), "%m-%d_%H-%M")
    file_path = os.path.join(MODEL_PATH, "lstm_{}_{:.4f}".format(timestamp, accuracy))
    torch.save(lstm, file_path)
