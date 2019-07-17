import torch
from torch import nn


class Block(nn.Module):

    def __init__(self, channel):
        super(Block, self).__init__()

        hidden_channel = channel // 4
        self.pre = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.conv1x1 = nn.Conv2d(channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv3x3_d1 = nn.Conv2d(channel, hidden_channel, kernel_size=3, stride=1, padding=1)
        self.conv3x3_d2 = nn.Conv2d(channel, hidden_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3x3_d3 = nn.Conv2d(channel, hidden_channel, kernel_size=3, stride=1, padding=3, dilation=3)

        self.final1x1 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x_ = self.pre(x)
        out1 = self.conv1x1(x_)
        out2 = self.conv3x3_d1(x_)
        out3 = self.conv3x3_d2(x_)
        out4 = self.conv3x3_d3(x_)
        out = self.final1x1(torch.cat([out1, out2, out3, out4], dim=1))
        out += x

        return out


class Attention(nn.Module):

    def __init__(self, channel):
        super(Attention, self).__init__()

        self.c_attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.c_attention2 = nn.Sequential(
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.LR_attention = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.avg_pooling = nn.AvgPool2d(16, 1)

    def forward(self, L, R):
        hidden_L = self.c_attention1(L)
        hidden_R = self.c_attention1(R)
        hidden = torch.cat([hidden_L, hidden_R], dim=1)
        L = L * self.c_attention2(hidden_L)
        R = R * self.c_attention2(hidden_R)
        hidden = self.LR_attention(hidden)
        L = L * hidden
        R = R * (1 - hidden)

        return self.avg_pooling(L + R)


class FusionNet(nn.Module):

    whole_channel = 24

    def __init__(self):
        super(FusionNet, self).__init__()

        channel = FusionNet.whole_channel
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
            nn.Conv2d(channel, channel, 3, 1, 1),

            Block(channel),
            Block(channel)
        )
        self.block2 = nn.Sequential(
            Block(channel),
            Block(channel)
        )
        self.block3 = nn.Sequential(
            Block(channel),
            Block(channel)
        )

        self.l_att = Attention(channel)
        self.m_att = Attention(channel)
        self.h_att = Attention(channel)

        self.final_conv = nn.Conv2d(channel*3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, LRimg):
        L, R = LRimg[:, 0, ...], LRimg[:, 1, ...]

        L1 = self.block1(L)
        L2 = self.block2(L1)
        L3 = self.block3(L2)

        R1 = self.block1(R)
        R2 = self.block2(R1)
        R3 = self.block3(R2)

        Low_feature = self.l_att(L1, R1)
        mid_feature = self.m_att(L2, R2)
        hig_feature = self.h_att(L3, R3)

        feature = torch.cat([Low_feature, mid_feature, hig_feature], dim=1)

        return self.final_conv(feature)


# class FusionNet(nn.Module):
#
#     whole_channel = 64
#
#     def __init__(self):
#         super(FusionNet, self).__init__()
#
#         channel = FusionNet.whole_channel
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(2, 2),
#
#             nn.Conv2d(channel, channel, 3, 1, 1),
#         )
#         self.block2 = nn.Sequential(
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, 3, 1, 1)
#         )
#         self.block3 = nn.Sequential(
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, 3, 1, 1),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel, channel, 3, 1, 1)
#         )
#
#         self.l_att = Attention(channel)
#         self.m_att = Attention(channel)
#         self.h_att = Attention(channel)
#
#         self.final_conv = nn.Conv2d(channel*3, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, LRimg):
#         L, R = LRimg[:, 0, ...], LRimg[:, 1, ...]
#
#         L1 = self.block1(L)
#         L2 = self.block2(L1)
#         L3 = self.block3(L2)
#
#         R1 = self.block1(R)
#         R2 = self.block2(R1)
#         R3 = self.block3(R2)
#
#         Low_feature = self.l_att(L1, R1)
#         mid_feature = self.m_att(L2, R2)
#         hig_feature = self.h_att(L3, R3)
#
#         feature = torch.cat([Low_feature, mid_feature, hig_feature], dim=1)
#
#         return self.final_conv(feature)


if __name__ == '__main__':
    import torchsnooper
    from torchsummary import summary

    with torchsnooper.snoop():

        LRimg = torch.randn((2, 2, 3, 360, 640)).cuda()
        model = FusionNet().cuda()
        out = model(LRimg)
        print(out.shape)

        # hidden = torch.randn((128, 64, 32, 32)).cuda()
        # model = Block(64).cuda()
        # out = model(hidden)
        # print(out.shape)

    # model = FusionNet().cuda()
    # summary(model, (2, 3, 360, 640))

