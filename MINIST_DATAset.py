import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose(
    # totersor是为了归一化， nornalize是为了标准化（变成标准正态分布）
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set = datasets.MNIST(root="../dataset/mnist", transform=transform, train=True,
                           download=True)
test_set = datasets.MNIST(root="../dataset/mnist", transform=transform, train=False,
                          download=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# 测试集不用shuffle，这样可以看到每次一样的测试结果是怎么样的
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 32)
        self.l5 = torch.nn.Linear(32, 10)

    def forward(self, x):
        # 将输入的数据转成 n*784的，-1表示n为自动计算
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # 最后一个不用做非线性转换
        x = self.l5(x)
        return x


net = Net()
criterion = torch.nn.CrossEntropyLoss()
# momentum冲量值 ，就是adam 里面的那个
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.5)


def train(data):
    for item in data:
        x, labels = item
        y = net(x)
        loss = criterion(y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(data):
    correct = 0
    total = 0
    with torch.no_grad():
        for item in data:
            x, labels = item
            y = net(x)
            # dim为0指的是每一行往下数，1是每行的列   每一行最大的列下标
            _, predicted = torch.max(y.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(correct / (total * 1.0))


for i in range(100):
    train(train_loader)
    test(test_loader)
