import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10
        # full connected layer
        # activation function
        # full connected layer
        # ......
        self.network = nn.Sequential(
            nn.Linear(in_features=32 * 32 * 3, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        return self.network(x)


def train(model, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    writer = SummaryWriter(log_dir=f'runs/{args.model}_{args.optimizer}_{args.scheduler}')
    criterion = nn.CrossEntropyLoss()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 128
    # create dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    # create dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    best_val_accuracy = 0.0
    # for-loop
    for epoch in range(60):
        # train
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss backward
            loss.backward()
            # optimize
            optimizer.step()
            # print statistics
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        avg_epoch_train_loss = epoch_train_loss / len(trainloader)
        train_accuracy = 100 * train_correct / train_total
        writer.add_scalar('Training loss (per epoch)', avg_epoch_train_loss, epoch)
        writer.add_scalar('Training Accuracy (per epoch)', train_accuracy, epoch)
        # adjust learning rate
        scheduler.step()

        # test
        model.eval()
        val_correct = 0
        val_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in valloader:
                # forward
                inputs, labels = data
                outputs = model(inputs)
                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
                'args': vars(args)
            }
            torch.save(checkpoint, 'best_checkpoint_fcnn.pth')
    writer.close()


def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    checkpoint = torch.load('best_checkpoint_fcnn.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # create testing dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # create dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    # test
    model.eval()
    test_correct = 0
    test_total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # forward
            inputs, labels = data
            outputs = model(inputs)
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier(3 * 32 * 32, 10)
    elif args.model == 'fcnn':
        model = FCNN(3 * 32 * 32, 256, 10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else:
        raise AssertionError
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=cosine
# python main.py --run=test --model=fcnn --optimizer=adamw --scheduler=cosine
