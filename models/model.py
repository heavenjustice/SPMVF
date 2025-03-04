import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                               bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                               bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        # Ensure the dimensions match for addition of residual
        self.match_dimensions = nn.Sequential()
        if in_channels != out_channels:
            self.match_dimensions = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                              padding=(0, 0, 0))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        residual = self.match_dimensions(residual)  # Match dimensions if needed
        out += residual  # Add the residual to the output
        out = self.relu(out)
        return out

class WideBranchNet_R(nn.Module):

    def __init__(self, time_length=7, num_classes=81):
        super(WideBranchNet_R, self).__init__()
        
        self.time_length = time_length
        self.num_classes = num_classes

        self.model = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            ResidualBlock(32, 64),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            ResidualBlock(64, 64),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
            )
        self.max2d = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )
        
    
    def forward(self, x):
        out = self.model(x)

        #residual = x
        #if residual.size(2) != out.size(2) or residual.size(3) != out.size(3) or residual.size(4) != out.size(4):
        #   residual = F.interpolate(residual, size=out.size()[2:], mode='trilinear', align_corners=True)
        #out += residual

        out = out.squeeze(2)
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))

        out = self.classifier(out)
        return out


if __name__ == '__main__':
    net = WideBranchNet_R(time_length=7, num_classes= 81)
    x = torch.rand(2, 3, 7, 64, 64)
    out = net(x)
