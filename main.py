# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#
# train_set = torchvision.datasets.MNIST(root="../dataset/mnist", train=False, download=True)
# test_set = torchvision.datasets.MNIST(root="../dataset/mnist", train=True, download=True)


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.length = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          )


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.liner1 = torch.nn.Linear(8, 5)
        self.liner2 = torch.nn.Linear(5, 4)
        self.liner3 = torch.nn.Linear(4, 2)
        self.liner4 = torch.nn.Linear(2, 1)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmod(self.liner1(x))
        x = self.sigmod(self.liner2(x))
        x = self.sigmod(self.liner3(x))
        x = self.sigmod(self.liner4(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for i in range(10000):
    for _, data in enumerate(train_loader):
        inputs, labels = data
        y = model(inputs)
        loss = criterion(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
