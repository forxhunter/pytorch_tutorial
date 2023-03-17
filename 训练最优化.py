import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 超参数 hyper para
batch_size = 64
learning_rate = 1e-3
epochs = 5
# 定义训练集和测试集
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root= "data",
    train = False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

# 定义神经网络
class MyNetwork(nn.Module):
    def __init__(self) -> None:
        super(MyNetwork,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512)
            nn.ReLU(),
            nn.Linear(512, 10),

        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#实例化
model = MyNetwork()
