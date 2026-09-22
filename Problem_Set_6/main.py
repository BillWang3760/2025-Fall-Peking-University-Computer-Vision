import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from models import VGG, ResNet, ResNext
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train(model, args):
    '''
    Model training function
    input:
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # create dataset, data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    # create dataloader
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30)
    # create summary writer
    writer = SummaryWriter(log_dir=f'/kaggle/working/runs/{args.model}_{args.optimizer}_{args.scheduler}')

    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0.0
    for epoch in range(0, 30):
        # train
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        avg_epoch_train_loss = epoch_train_loss / len(trainloader)
        train_accuracy = 100 * train_correct / train_total
        writer.add_scalar('Training loss (per epoch)', avg_epoch_train_loss, epoch)
        writer.add_scalar('Training Accuracy (per epoch)', train_accuracy, epoch)
        # scheduler adjusts learning rate
        scheduler.step()

        # test
        model.eval()
        val_correct = 0
        val_total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for inputs, labels in valloader:
                # forward
                inputs, labels = inputs.to(device), labels.to(device)
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
            torch.save(checkpoint, f'/kaggle/working/checkpoint_{args.model}.pth')
    writer.close()


def test(model, args):
    '''
    input:
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    checkpoint = torch.load(f'checkpoint_{args.model}.pth', map_location=torch.device('cpu'))
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
        for inputs, labels in testloader:
            # forward
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy} %')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='vgg')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'vgg':
        model = VGG()
    elif args.model == 'resnet':
        model = ResNet()
    elif args.model == 'resnext':
        model = ResNext()
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, args)
    elif args.run == 'test':
        test(model, args)
    else:
        raise AssertionError

# python main.py --run=train --model=vgg --optimizer=adamw --scheduler=cosine
# python main.py --run=test --model=vgg --optimizer=adamw --scheduler=cosine
