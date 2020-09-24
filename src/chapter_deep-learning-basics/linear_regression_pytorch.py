import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch import nn
from torch.nn import init


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


def main():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)).astype(np.float32))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    # net = LinearNet(num_inputs)
    net = nn.Sequential(nn.Linear(num_inputs, 1))

    # 初始化模型参数
    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化算法
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))

    dense = net[0]
    print(true_w, dense.weight)
    print(true_b, dense.bias)


if __name__ == '__main__':
    main()
