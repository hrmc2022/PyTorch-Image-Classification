import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    '''ResNet18 Residual Block
    Args:
        in_channels(int): input channels
        out_channels(int): output channels
        stride(int): stride    
    '''
    def __init__(self, in_cannnels: int, out_channnels: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_cannnels, out_channnels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channnels)
        self.conv2 = nn.Conv2d(out_channnels, out_channnels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channnels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_cannnels, out_channnels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channnels)
            )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out  = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return self.relu(x + out)
        

class ResNet18(nn.Module):
    '''ResNet18
    Args:
        num_classes(int): the number of classes
    '''
    def __init__(self, num_classes: int):
        super().__init__()
        # Size: (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Size: (64, 16, 16)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Size: (64, 8, 8)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )
        # Size: (512, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.flatten(1)
        
        if return_embed:
            return x
        
        x = self.linear(x)
        
        return x