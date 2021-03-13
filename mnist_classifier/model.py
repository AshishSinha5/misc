import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import mnistDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse

class linear_nn(nn.Module):

    def __init__(self):
        super(linear_nn, self).__init__()
        self.dense1 = nn.Linear(28 * 28, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.dense3 = nn.Linear(128, 64)
        self.op = nn.Linear(64, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = F.tanh(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.op(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description='pytorch mnist classification')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size for training')
    parser.add_argument("--test_batch_size", type=int, default=1000, help='batch size for testing')
    parser.add_argument("--epochs", type=int, default=14, help='number of epochs for training')
    parser.add_argument("--lr", type=float, default=0.6, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="learning rate step")
    parser.add_argument("--no_cuda", action='store_true', default=False, help="disables CUDA training")
    parser.add_argument("--dry_run", action='store_true', default=False, help="train for single pass")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="how many batches to wait before logging "
                                                                     "training status")
    parser.add_argument("--save_model", action='store_true', default=True, help='save model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    print(args)

    dataset1 = mnistDataset(file_path='data/mnist_train.csv')
    dataset2 = mnistDataset(file_path='data/mnist_test.csv')
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = linear_nn().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_acc = []
    test_loss = []
    for epoch in range(1, args.epochs + 1):
        train_model(args, model, device, train_loader, optimizer, epoch)
        loss, acc = test_model(model, device, test_loader)
        test_loss.append(loss)
        test_acc.append(acc)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "output/mnist_cnn.pt")

    plt.plot(range(1, args.epochs+1), test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/loss.png')
    plt.close()

    plt.plot(range(1, args.epochs+1), test_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/acc.png')
    plt.close()


if __name__ == '__main__':
    main()
