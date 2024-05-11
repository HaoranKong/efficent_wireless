import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 全连接层
        self.fc1 = nn.Linear(20 * 4 * 4, 320)  # 计算输出特征图的大小为 4x4
        self.fc2 = nn.Linear(320, 80)
        # 输出层
        self.fc3 = nn.Linear(80, 10)  # 10 个类别的输出

    def forward(self, x):
        # 卷积 -> 池化 -> 激活函数
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 将特征图展平
        x = x.view(-1, 20 * 4 * 4)
        # 全连接层 -> 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x