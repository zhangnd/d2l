import d2lzh_pytorch as d2l
import numpy as np
import torch

num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float32, requires_grad=True)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def main():
    batch_size = 256
    root = '../Datasets'
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root=root)

    num_epochs, lr = 5, 0.1

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    X, y = iter(test_iter).next()

    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == '__main__':
    main()
