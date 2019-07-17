import torch
from torch import nn

channel = 64

class Proposed(nn.Module):

    def __init__(self):
        super(Proposed, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 30*30*32
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 14*14*32
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 5*5*32
        )

        self.avg1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(30, 30)
        )
        self.avg2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(14, 14)
        )
        self.avg3 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(5, 5)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(channel * 3, channel),
            nn.PReLU(),
            nn.Dropout()
        )
        self.weight = nn.Linear(channel, 1)
        self.score = nn.Sequential(
            nn.Linear(channel, 8),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(8, 1)
        )

    def forward(self, imgL, imgR):
        bs = imgL.size(0)
        xL = self.down1(imgL)
        xL_side1 = self.avg1(xL).view(bs, -1)
        xR = self.down1(imgR)
        xR_side1 = self.avg1(xR).view(bs, -1)

        xL = self.down2(xL)
        xL_side2 = self.avg2(xL).view(bs, -1)
        xR = self.down2(xR)
        xR_side2 = self.avg2(xR).view(bs, -1)
        xL = self.down3(xL)
        xL_side3 = self.avg3(xL).view(bs, -1)
        xR = self.down3(xR)
        xR_side3 = self.avg3(xR).view(bs, -1)

        xL = torch.cat([xL_side1, xL_side2, xL_side3], dim=1)
        xR = torch.cat([xR_side1, xR_side2, xR_side3], dim=1)

        xL = self.linear1(xL)
        wL = self.weight(xL)
        sL = self.score(xL)

        xR = self.linear1(xR)
        wR = self.weight(xR)
        sR = self.score(xR)

        wL, wR = torch.exp(wL), torch.exp(wR)
        s = (wL * sL + wR * sR) / (wL + wR)

        return s.squeeze()


class Proposed2(nn.Module):

    def __init__(self):
        super(Proposed2, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 30*30*32
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 14*14*32
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 3, 1, 0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),  # 5*5*32
        )

        self.avg1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(30, 30)
        )
        self.avg2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(14, 14)
        )
        self.avg3 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.AvgPool2d(5, 5)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(channel * 3, channel),
            nn.PReLU(),
            nn.Dropout()
        )
        self.weight = nn.Linear(channel, 1)
        self.score = nn.Sequential(
            nn.Linear(channel, 8),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(8, 1)
        )

    def forward(self, imgL, imgR):
        bs = imgL.size(0)
        xL = self.down1(imgL)
        xL_side1 = self.avg1(xL).view(bs, -1)
        xR = self.down1(imgR)
        xR_side1 = self.avg1(xR).view(bs, -1)

        xL = self.down2(xL)
        xL_side2 = self.avg2(xL).view(bs, -1)
        xR = self.down2(xR)
        xR_side2 = self.avg2(xR).view(bs, -1)
        xL = self.down3(xL)
        xL_side3 = self.avg3(xL).view(bs, -1)
        xR = self.down3(xR)
        xR_side3 = self.avg3(xR).view(bs, -1)

        xL = torch.cat([xL_side1, xL_side2, xL_side3], dim=1)
        xR = torch.cat([xR_side1, xR_side2, xR_side3], dim=1)

        xL = self.linear1(xL)
        wL = self.weight(xL)
        sL = self.score(xL)

        xR = self.linear1(xR)
        wR = self.weight(xR)
        sR = self.score(xR)

        wL, wR = torch.exp(wL), torch.exp(wR)
        s = (wL * sL + wR * sR) / (wL + wR)

        wL_pre = wL / (wL + wR)
        wR_pre = wR / (wL + wR)

        return s.squeeze(), wL_pre.squeeze(), wR_pre.squeeze()