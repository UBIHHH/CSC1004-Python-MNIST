from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import warnings

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    train the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    training_acc, training_loss = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        training_acc += (output.argmax(dim=1) == target).type(torch.float).sum().item()
        training_loss += loss.item()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    training_acc /= len(train_loader.dataset)
    training_loss /= len(train_loader)

    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    testing_acc = 0
    testing_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_pred = model(data)
            loss = F.nll_loss(target_pred, target)
            testing_loss += loss.item()
            testing_acc += (target_pred.argmax(dim=1) == target).type(torch.float).sum().item()

    testing_acc /= len(test_loader.dataset)
    testing_loss /= len(test_loader)
    return testing_acc, testing_loss


def plot_training_acc(epoches, training_accuracies, seed):
    warnings.filterwarnings("ignore")

    epoches_range = range(1, len(epoches) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, training_accuracies, 's-', color='r', label='training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title(f'Training accuracy (seed {seed})')
    plt.savefig(f'training_accuracy_seed_{seed}.png')


def plot_training_loss(epoches, training_loss, seed):
    warnings.filterwarnings("ignore")
    epoches_range = range(1, len(epoches) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, training_loss, 's-', color='r', label='training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Training loss (seed {seed})')
    plt.savefig(f'training_loss_seed_{seed}.png')


def plot_testing_acc(epoches, testing_accuracies, seed):
    warnings.filterwarnings("ignore")
    epoches_range = range(1, len(epoches) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, testing_accuracies, 's-', color='r', label='test accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title(f'Testing accuracies (seed {seed})')
    plt.savefig(f'testing_acc_seed_{seed}.png')


def plot_testing_loss(epoches, testing_loss, seed):
    warnings.filterwarnings("ignore")
    epoches_range = range(1, len(epoches) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, testing_loss, 's-', color='r', label='training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Testing loss (seed {seed})')
    plt.savefig(f'testing_loss_seed_{seed}.png')


def run(rank, config, seed):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)
    print(f"Process {rank} using seed {seed}")
    if use_cuda:
        device = torch.device("cuda", rank)
    elif use_mps:
        device = torch.device("mps", rank)
    else:
        device = torch.device("cpu", rank)

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    # add random seed to the DataLoader
    torch.manual_seed(seed)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        """record training info, Fill your code"""
        test_acc, test_loss = test(model, device, test_loader)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        with open(f"epoch_{epoch}_rank_{rank}_seed_{seed}.txt", "w") as f:
            f.write(f"Training accuracy: {train_acc}\n")
            f.write(f"Training loss: {train_loss}\n")
            f.write(f"Testing accuracy: {test_acc}\n")
            f.write(f"Testing loss: {test_loss}\n")
        """record testing info, Fill your code"""
        epoches.append(epoch)
        scheduler.step()
        """update the records, Fill your code"""

    """plotting training performance with the records"""
    plot_training_acc(epoches, training_accuracies, seed)
    plot_training_loss(epoches, training_loss, seed)

    """plotting testing performance with the records"""
    plot_testing_acc(epoches, testing_accuracies, seed)
    plot_testing_loss(epoches, testing_loss, seed)

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean(epoches):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""

    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    for epoch in range(1, config.epochs + 1):
        train_acc_sum = 0
        train_loss_sum = 0
        test_acc_sum = 0
        test_loss_sum = 0

        for seed in [123, 321, 666]:
            with open(f"epoch_{epoch}_rank_0_seed_{seed}.txt", "r") as f:
                lines = f.readlines()
                train_acc_sum += float(lines[0].split(": ")[1])
                train_loss_sum += float(lines[1].split(": ")[1])
                test_acc_sum += float(lines[2].split(": ")[1])
                test_loss_sum += float(lines[3].split(": ")[1])

        train_acc_mean = train_acc_sum / 3
        train_loss_mean = train_loss_sum / 3
        test_acc_mean = test_acc_sum / 3
        test_loss_mean = test_loss_sum / 3

        training_accuracies.append(train_acc_mean)
        training_loss.append(train_loss_mean)
        testing_accuracies.append(test_acc_mean)
        testing_loss.append(test_loss_mean)

    warnings.filterwarnings("ignore")

    epoches_range = range(1, epoches + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, training_accuracies, 's-', color='r', label='training accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title(f'Mean Training accuracy')
    plt.savefig(f'mean_training_accuracy.png')

    warnings.filterwarnings("ignore")

    epoches_range = range(1, epoches + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, training_loss, 's-', color='r', label='training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Mean Training loss')
    plt.savefig(f'mean_training_loss.png')

    warnings.filterwarnings("ignore")

    epoches_range = range(1, epoches + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, testing_accuracies, 's-',color='r', label='test accuracies')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title(f'Mean Testing accuracies')
    plt.savefig(f'mean_testing_acc.png')

    warnings.filterwarnings("ignore")

    epoches_range = range(1, epoches + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epoches_range, testing_loss, 's-', color='r', label='training loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(f'Mean Testing loss')
    plt.savefig(f'mean_testing_loss.png')


if __name__ == '__main__':
    arg = read_args()
    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    torch.multiprocessing.spawn(run, args=(config, 123), nprocs=3, join=True)
    torch.multiprocessing.spawn(run, args=(config, 321), nprocs=3, join=True)
    torch.multiprocessing.spawn(run, args=(config, 666), nprocs=3, join=True)

    """plot the mean results"""
    plot_mean(config.epochs)
