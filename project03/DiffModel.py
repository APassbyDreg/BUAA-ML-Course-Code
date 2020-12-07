import numpy as np

import torch
from torch import nn
from torch.nn import functional as func
from torchvision import transforms


# setting up accel device
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")


MAX_TRY = 5


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DiffModel(nn.Module):
    def __init__(self):
        # initing model
        self.version = "v1"
        super(DiffModel, self).__init__()
        # set layers
        self.input = nn.Linear(2000, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc1 = nn.Linear(1000, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 40)
        self.bn4 = nn.BatchNorm1d(40)
        self.output = nn.Linear(40, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        assert(self.version == "v1")

        x = torch.cat([x1, x2], dim=1)
        x = self.input(x)   
        x = func.relu(self.fc1(self.bn1(x)))
        x = func.relu(self.fc2(self.bn2(x)))
        x = func.relu(self.fc3(self.bn3(x)))
        return torch.sigmoid(self.output(self.bn4(x)))


class DiffModelDataset(torch.utils.data.Dataset):
    def __init__(self, ds, n_p_ratio=1, extend=1):
        """
            ds: size=(n_class, n_sample, len(data)), ds[i] represents all data in class i
            n_p_ratio: n_negative / n_positive, integer
        """
        self.ds = ds
        self.num_classes = len(ds)
        self.n_p_ratio = n_p_ratio
        self.extend = extend
        self.length = len(ds) * len(ds[0]) * (1 + self.n_p_ratio)
        # used to go through and generate data
        data_indexs = []
        for c in range(len(ds)):
            for i in range(len(ds[c])):
                data_indexs.append([c, i])
        self.data_idx = np.array(data_indexs)
        self.data_loaded = 0
        self.waitlist = []    

    def __len__(self):
        return self.length * self.extend

    def __getitem__(self, idx):
        with torch.no_grad():
            # class_idx and sample_idx of x1 is specified by idx
            c1, i1 = self.data_idx[(idx % self.length) // (self.n_p_ratio + 1)]
            # class_idx of x2 is specified by idx and random number
            c2 = c1
            if idx % (self.n_p_ratio + 1) != 0:
                y = torch.tensor([0], dtype=torch.float, requires_grad=False)
                n_try = 0
                while c2 == c1 and n_try < MAX_TRY:
                    c2 = np.random.randint(self.num_classes)
            else:
                y = torch.tensor([1], dtype=torch.float, requires_grad=False)
            # sample_idx of x2 is a random number
            i2 = np.random.randint(len(self.ds[c2]))
            # get x1, x2 as tensors
            x1 = torch.tensor(self.ds[c1][i1], dtype=torch.float)
            x2 = torch.tensor(self.ds[c2][i2], dtype=torch.float)
        return x1, x2, y
            

def img2input(img):
    t1 = transforms.ToTensor(),
    t2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                              0.229, 0.224, 0.225]),
    tensor = t1[0](img)
    if tensor.size()[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = t2[0](tensor)
    tensor = tensor.view(1, tensor.size()[0], tensor.size()[1], tensor.size()[2])
    tensor.requires_grad = False
    resnet = torch.load(r"models\pretrained\resnet50_pretrained")
    return resnet(tensor).tolist()[0]


def predict(train_ds, input, model):
    if len(input) != len(train_ds[0][0]):
        raise ValueError("invalid input, use function 'img2input' before predicting")

    model.to(dev)
    p = np.zeros(len(train_ds))
    input = torch.tensor([input], requires_grad=False).to(dev)

    model.eval()
    for group in range(len(train_ds)):
        for sample in range(len(train_ds[group])):
            truth = torch.tensor([train_ds[group][sample]],requires_grad=False).to(dev)
            with torch.no_grad():
                p[group] += (model(input, truth) + model(truth, input)).item()

    selection = np.argmax(p)
    return selection, p[selection] / (len(train_ds[selection]) * 2)
