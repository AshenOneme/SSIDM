import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个 Inception 模块
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        # 拼接所有分支
        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)  # 在通道维度上拼接
        return outputs

# 定义 Inception V4 类
class InceptionV4(nn.Module):
    def __init__(self, input_channels=1, num_classes=65):
        super(InceptionV4, self).__init__()
        # 输入层
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2),  # 输出: 32通道
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 输出: 64通道
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1),  # 输出: 96通道
            nn.ReLU()
        )

        # 堆叠多个 Inception 模块
        self.inception_a = InceptionModule(96, 16)  # 输入 96 通道，输出 64 通道
        self.inception_b = InceptionModule(64, 32)  # 输入 256 通道，输出 128 通道
        self.inception_c = InceptionModule(128, 64)  # 输入 512 通道，输出 256 通道

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出固定为 1x1 的特征图

        # 全连接层
        self.fc = nn.Linear(256, num_classes)  # 输入 1024 个特征，输出 65 类

    def forward(self, x):
        x = self.stem(x)  # 输入层
        x = self.inception_a(x)  # 第一个 Inception 模块
        x = self.inception_b(x)  # 第二个 Inception 模块
        x = self.inception_c(x)  # 第三个 Inception 模块
        x = self.global_avg_pool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 拉平为二维张量
        x = self.fc(x)  # 全连接层
        return x

# # 实例化模型
# model = InceptionV4(input_channels=1, num_classes=65)
#
# # 打印模型结构
# print(model)
#
# # 测试模型
# input_tensor = torch.randn(1, 1, 240, 480)  # 单张图片，1 通道，尺寸为 240x480
# output = model(input_tensor)
# print(output.shape)  # 输出应为 [1, 65]
