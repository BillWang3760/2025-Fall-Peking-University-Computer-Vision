import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        self.features = nn.Sequential(
            # Stage 1: conv-conv-pool
            self.conv_block(3, 64),
            self.conv_block(64, 64),
            nn.MaxPool2d(2, 2),  # -> N * 64 * 16 * 16
            # Stage 2: conv-conv-pool
            self.conv_block(64, 128),
            self.conv_block(128, 128),
            nn.MaxPool2d(2, 2),  # -> N * 128 * 8 * 8
            # Stage 3: conv-conv-pool
            self.conv_block(128, 256),
            self.conv_block(256, 256),
            nn.MaxPool2d(2, 2),  # -> N * 256 * 4 * 4
            # Stage 4: conv-conv-conv-pool
            self.conv_block(256, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            nn.MaxPool2d(2, 2),  # -> N * 512 * 2 * 2
            # Stage 5: conv-conv-conv-pool
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            nn.MaxPool2d(2, 2),  # -> N * 512 * 1 * 1
        )
        # 2. define full-connected layer to classify
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10)
        )

    def conv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class ResBlock(nn.Module):
    ''' residual block'''

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        # 1. define double convolution
        # convolution
        # batch normalization
        # activate function
        # ......
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        self.conv3 = None
        if in_channel != out_channel or stride != 1:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H * W]
        # 1. convolve the input
        out = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        if self.in_channel != self.out_channel or self.stride != 1:
            x = self.conv3(x)
        # 3. Add the output of the convolution and the original data (or from 2.)
        out += x
        # 4. relu
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    '''residual network'''

    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 2. define multiple residual blocks
        self.resnet_block_1 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.resnet_block_2 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.resnet_block_3 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.resnet_block_4 = nn.Sequential(*self.resnet_block(256, 512, 2))
        # 3. define full-connected layer to classify
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def resnet_block(self, in_channel, out_channel, num_ResBlocks, first_block=False):
        resnet_block = []
        for i in range(num_ResBlocks):
            if i == 0 and not first_block:
                resnet_block.append(ResBlock(in_channel, out_channel, stride=2))
            else:
                resnet_block.append(ResBlock(out_channel, out_channel, stride=1))
        return resnet_block

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv(x)
        x = self.resnet_block_4(self.resnet_block_3(self.resnet_block_2(self.resnet_block_1(x))))
        # classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ResNextBlock(nn.Module):
    '''ResNext block'''

    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.
        hidden_channel = out_channel // bottle_neck
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        # 1. define convolution
        # 1x1 convolution
        # batch normalization
        # activate function
        # 3x3 convolution
        # ......
        # 1x1 convolution
        # ......
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=group)
        self.bn2 = nn.BatchNorm2d(hidden_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        self.conv4 = None
        if in_channel != out_channel or stride != 1:
            self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input.
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution
        # 3. Add the output of the convolution and the original data (or from 2.)
        if self.in_channel != self.out_channel or self.stride != 1:
            out += self.conv4(x)
        else:
            out += x
        # 4. relu
        out = self.relu3(out)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 2. define multiple residual blocks
        self.resnext_layer_1 = nn.Sequential(*self.resnext_layer(64, 64, 2, 2, 32, first_block=True))
        self.resnext_layer_2 = nn.Sequential(*self.resnext_layer(64, 128, 2, 2, 32))
        self.resnext_layer_3 = nn.Sequential(*self.resnext_layer(128, 256, 2, 2, 32))
        self.resnext_layer_4 = nn.Sequential(*self.resnext_layer(256, 512, 2, 2, 32))
        # 3. define full-connected layer to classify
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def resnext_layer(self, in_channel, out_channel, num_blocks, bottle_neck, group, first_block=False):
        resnext_layer = []
        for i in range(num_blocks):
            if i == 0 and not first_block:
                resnext_layer.append(ResNextBlock(in_channel, out_channel, bottle_neck, group, stride=2))
            else:
                resnext_layer.append(ResNextBlock(out_channel, out_channel, bottle_neck, group, stride=1))
        return resnext_layer

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        x = self.conv(x)
        x = self.resnext_layer_4(self.resnext_layer_3(self.resnext_layer_2(self.resnext_layer_1(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
