import torch
import torch.nn as nn
import torch.nn.functional as F

'''-------------一、BasicBlock模块-----------------------------'''


# 用于ResNet18和ResNet34基本残差结构块
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.downsample(downsample)
        )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.left(x)  # 这是由于残差块需要保留原始输入
        out += identity  # 这是ResNet的核心，在输出上叠加了输入x
        out = F.relu(out)
        return out


'''-------------二、Bottleneck模块-----------------------------'''


class Bottleneck(nn.Module):
    expansion = 4

    # 这里相对于RseNet，在代码中增加一下两个参数groups和width_per_group（即为group数和conv2中组卷积每个group的卷积核个数）
    # 默认值就是正常的ResNet
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        # 这里也可以自动计算中间的通道数，也就是3x3卷积后的通道数，如果不改变就是out_channels
        # 如果groups=32,with_per_group=4,out_channels就翻倍了
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        # 组卷积的数，需要传入参数
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # -----------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


'''-------------三、搭建ResNeXt结构-----------------------------'''


class ResNeXt(nn.Module):
    def __init__(self,
                 block,  # 表示block的类型
                 blocks_num,  # 表示的是每一层block的个数
                 num_classes=1000,  # 表示类别
                 include_top=True,  # 表示是否含有分类层(可做迁移学习)
                 groups=1,  # 表示组卷积的数
                 width_per_group=64):
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 64 -> 128
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 128 -> 256
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 256 -> 512
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 512 -> 1024
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    # 形成单个Stage的网络结构
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        # 该部分是将每个blocks的第一个残差结构保存在layers列表中。
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion  # 得到最后的输出

        # 该部分是将每个blocks的剩下残差结构保存在layers列表中，这样就完成了一个blocks的构造。
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def ResNet34(num_classes=1000, include_top=True):
    return ResNeXt(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet50(num_classes=1000, include_top=True):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def ResNet101(num_classes=1000, include_top=True):
    return ResNeXt(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


# 论文中的ResNeXt50_32x4d
def ResNeXt50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNeXt(Bottleneck, [3, 4, 6, 3],
                   num_classes=num_classes,
                   include_top=include_top,
                   groups=groups,
                   width_per_group=width_per_group)


def ResNeXt101_32x8d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNeXt(Bottleneck, [3, 4, 23, 3],
                   num_classes=num_classes,
                   include_top=include_top,
                   groups=groups,
                   width_per_group=width_per_group)


'''
if __name__ == '__main__':
    model = ResNeXt50_32x4d()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
# test()
'''
from torchsummary import summary

if __name__ == '__main__':
    net = ResNeXt50_32x4d().cuda()
    summary(net, (3, 224, 224))