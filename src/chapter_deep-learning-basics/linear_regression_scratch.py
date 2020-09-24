import random

# import matplotlib.pyplot as plt
import numpy as np
import torch


# 遍历数据集并不断读取小批量数据样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
# 小批量随机梯度下降算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def main():
    # 构造一个简单的人工训练数据集
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)).astype(np.float32))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))  # 噪声项ε服从均值为0、标准差为0.01的正太分布

    # plt.rcParams['figure.figsize'] = (3.5, 2.5)
    # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    # plt.show()

    # 初始化模型参数
    # 将权重初始化成均值为0、标准差为0.01的正太随机数，偏差则初始化为0
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    batch_size = 10
    lr = 0.03  # 学习率
    num_epochs = 3  # 迭代周期
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

            # 不要忘了梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)


if __name__ == '__main__':
    main()
