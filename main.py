import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, TensorDataset

from src.dataset.mnist import *
from src.models.cnn import CNN

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 创建模型实例
    model = CNN()

    # 打印模型结构
    print(model)

    (train_images, train_labels), (test_images, test_labels) = load_mnist(flatten=False)


    train_dataset = TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)



    test_dataset = TensorDataset(torch.tensor(test_images), torch.tensor(test_labels))

    # 使用 DataLoader 加载测试数据集
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            # 在使用损失函数之前将 labels 转换为 Long 类型
            labels = labels.long()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training finished.')

    model.eval()

    # 定义正确预测的数量和总样本数量
    correct = 0
    total = 0

    # 不计算梯度
    with torch.no_grad():
        for images, labels in test_loader:
            # 获取模型的预测结果
            outputs = model(images)
            # 选择具有最高概率的类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 更新总样本数量
            total += labels.size(0)
            # 计算正确预测的数量
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
