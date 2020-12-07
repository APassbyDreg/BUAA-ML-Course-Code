import torch
from torch import nn, optim
import torch.nn.functional as nnfunc

import os
from datetime import datetime

import Loader as myds


torch.device("cuda")
N_ITER = 600
MODEL_PATH = ".\\models\\train_from_extended"


class DNN(nn.Module):
    """
        a simple 4 layer DNN implement
    """

    def __init__(self, n_input, n_output):
        super(DNN, self).__init__()
        self.n_input = n_input
        self.fc1 = nn.Linear(n_input, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, n_output)

    def forward(self, x):
        x = nnfunc.relu(self.fc1(x))
        x = nnfunc.relu(self.fc2(x))
        x = nnfunc.relu(self.fc3(x))
        y = nnfunc.softmax(self.fc4(x), dim=-1)
        return y


if __name__ == "__main__":
    # prepare
    dnn = DNN(1200, 8)
    loss_func = nn.MSELoss()
    optimizer = optim.RMSprop(dnn.parameters(), lr=4e-5, momentum=0.8)
    lr_decay = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 700], gamma=0.75)

    # load datasets
    x, y = myds.load_dataset("train", require_flat=True, use_extended=True)
    test_x, test_y = myds.load_dataset("test", require_flat=True, use_extended=True)
    
    # train
    for i in range(N_ITER):
        dnn.zero_grad()
        out = dnn(x)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        lr_decay.step()
        # show
        if i % 10 == 9:
            with torch.no_grad():
                predict = dnn(test_x)
                test_loss = loss_func(predict ,test_y)
                print("iter {:03d}: loss={:.6f}, test_set_loss={:.6f}".format(i+1, loss.item(), test_loss.item()))

    # test accuracy
    with torch.no_grad():
        predict = dnn(test_x)
        correct = 0
        for i in range(test_x.size()[0]):
            if torch.argmax(predict[i], dim=-1) == torch.argmax(test_y[i], dim=-1):
                correct += 1
        accuracy = correct / test_x.size()[0]
        print("accuracy is {:.4f} ({}/{})".format(accuracy, correct, test_x.size()[0]))

    # save model
    timestamp = datetime.strftime(datetime.now(), "%m-%d_%H-%M")
    file_path = os.path.join(MODEL_PATH, "dnn_{}_{:.4f}".format(timestamp, accuracy))
    torch.save(dnn, file_path)
