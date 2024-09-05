import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


# YOLOv3 Model Definition
class YOLOv3(nn.Module):
    def __init__(self, numClasses, numAnchors):
        self.numClasses = numClasses
        self.numAnchors = numAnchors
        super(YOLOv3, self).__init__()
        self.backbone = Darknet53()
        # Detection Heads
        self.layer1 = nn.Sequential(
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),

            ConvBlock(1024, 1024 // 2, kernelSize=1, stride=1, padding=0),
            ConvBlock(1024 // 2, 1024, kernelSize=3, stride=1, padding=1),

            ConvBlock(1024 , 1024//2, kernelSize=1, stride=1, padding=0),

        )
        self.outLayer1 = ScalePred(1024//2, numClasses, numAnchors)

        self.layer2 = nn.Sequential(
            ConvBlock(1024//2, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2),

        )


        self.layer3 = nn.Sequential(
            ConvBlock(768, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),

            ConvBlock(512, 512 // 2, kernelSize=1, stride=1, padding=0),
            ConvBlock(512//2, 512, kernelSize=3, stride=1, padding=1),
            ConvBlock(512, 512//2, kernelSize=1, stride=1, padding=0),
        )
        self.outLayer2 = ScalePred(512//2, numClasses, numAnchors)
        self.layer4 = nn.Sequential(
            ConvBlock(512//2, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2),
        )


        self.layer5 = nn.Sequential(
            ConvBlock(384, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),

            ConvBlock(256, 256 // 2, kernelSize=1, stride=1, padding=0),
            ConvBlock(256 // 2, 256, kernelSize=3, stride=1, padding=1),
            ConvBlock(256, 256 // 2, kernelSize=1, stride=1, padding=0),
            ScalePred(256 // 2, numClasses, numAnchors)
        )


    def forward(self, x):
        # Forward pass through the backbone
        x,routeConnections = self.backbone(x)

        x1 = self.layer1(x)  # Feature map from the deepest layer
        out1 = self.outLayer1(x1)

        x2 = self.layer2(x1)
        x2 = torch.cat([x2,routeConnections[-1]],dim=1)
        routeConnections.pop()

        x3 = self.layer3(x2)  # Feature map from the intermediate layer
        out2 = self.outLayer2(x3)

        x4 = self.layer4(x3)
        x4 = torch.cat([x4, routeConnections[-1]], dim=1)
        routeConnections.pop()

        out3 = self.layer5(x4)  # Feature map from the earliest layer


        return [out1,out2,out3]




# Convolutional Block used in YOLO
class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(outChannels)
        self.leakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyReLU(self.bn(self.conv(x)))


# Residual Block used in Darknet-53
class ResidualBlock(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.conv1 = ConvBlock(inChannels, inChannels // 2, kernelSize=1, stride=1, padding=0)
        self.conv2 = ConvBlock(inChannels // 2, inChannels, kernelSize=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x # Skip connection


# Darknet-53 Backbone Network
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        self.layer1 = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ResidualBlock(64),
            ConvBlock(64, 128, 3, 2, 1),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 256, 3, 2, 1),
        )
        self.layer2 = nn.Sequential(
            *[ResidualBlock(256) for _ in range(8)]
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, 512, 3, 2, 1),
        )
        self.layer4 = nn.Sequential(
            *[ResidualBlock(512) for _ in range(8)]
        )

        self.layer5 = nn.Sequential(
            ConvBlock(512, 1024, 3, 2, 1),
            *[ResidualBlock(1024) for _ in range(4)]
        )

    def forward(self, x):
        routeConnections = []
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        routeConnections.append(x2)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        routeConnections.append(x4)
        x5 = self.layer5(x4)

        # Return feature maps at different scales
        return x5,routeConnections


class ScalePred(nn.Module):
    def __init__(self, inChannels, numClasses ,numAnchors):
        super().__init__()
        self.pred = nn.Sequential(
            ConvBlock(inChannels, 2*inChannels, kernelSize=3, padding=1, stride=1) ,
            nn.Conv2d(2*inChannels, numAnchors*(numClasses+5), kernel_size=1)
        )
        self.numClasses = numClasses
        self.numAnchors = numAnchors


    def forward(self,x):
        x = self.pred(x)
        return (
            x.reshape(x.shape[0], self.numAnchors, self.numClasses+5, x.shape[2], x.shape[3])
            .permute(0,1,3,4,2)
        )










###################### testing ###############
# Dummy input
# dummy_input = torch.randn((2, 3, 416, 416))  # Batch size 2, 3 channels, 416x416 image
#
# # Forward pass
# model = YOLOv3(numClasses=1, numAnchors=3)
# outputs = model(dummy_input)
#
# # Print output shapes
# for i, out in enumerate(outputs):
#     print(f"Output {i+1} shape: {out.shape}")
#
