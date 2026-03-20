import torch
import torch.nn as nn
from LRCED import LRCED

class WCMF(nn.Module):
    def __init__(self, channel=256):
        super(WCMF, self).__init__()
        #定义用于处理RGB输入的卷积层序列
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),  # 1x1卷积
            nn.BatchNorm2d(channel),              # 批归一化
            nn.ReLU()                             # 激活函数
        )


        # 定义用于处理深度输入的卷积层序列
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),  # 1x1卷积
            nn.BatchNorm2d(channel),              # 批归一化
            nn.ReLU()                             # 激活函数
        )
        # 定义融合特征的卷积层序列
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 3, 1, 1),  # 3x3卷积
            nn.BatchNorm2d(channel),                 # 批归一化
            nn.ReLU()                                # 激活函数
        )
        # 定义输出权重的卷积层序列
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, 2, 3, 1, 1),         # 3x3卷积
            nn.BatchNorm2d(2),                      # 批归一化
            nn.ReLU()                               # 激活函数
        )
        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lrc = LRCED(3)

    def fusion(self, f1, f2,  f_vec):
        # 提取权重
        w1 = f_vec[:, 0, :, :].unsqueeze(1)  # 提取第一个通道的权重
        w2 = f_vec[:, 1, :, :].unsqueeze(1)  # 提取第二个通道的权重

        # 计算加权和
        out1 = (w1 * f1) + (w2 * f2)
        # 计算加权乘积
        out2 = (w1 * f1) * (w2 * f2)

        # 返回融合结果
        return out1 + out2

    def forward(self, rgb, depth):
        # 处理RGB输入
        rgb = self.lrc(rgb)
        depth = self.lrc(depth)
        Fr = self.conv_r1(rgb) #+ self.lrc(rgb)
        # 处理深度输入
        Fd = self.conv_d1(depth) #+ self.lrc(depth)

        # 特征拼接
        f = torch.cat([Fr, Fd], dim=1)
        # 融合特征
        f = self.conv_c1(f)
        # 计算权重
        f = self.conv_c2(f)

        # 进行特征融合
        Fo = self.fusion(Fr, Fd, f)
        return Fo
class many_WCMF(nn.Module):
    def __init__(self, channel=256):
        super(many_WCMF, self).__init__()
        self.wcm = WCMF(3)
        # 定义融合特征的卷积层序列
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 3, 1, 1),  # 3x3卷积
            nn.BatchNorm2d(channel),  # 批归一化
            nn.ReLU()  # 激活函数
        )
        # 定义输出权重的卷积层序列
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, 2, 3, 1, 1),  # 3x3卷积
            nn.BatchNorm2d(2),  # 批归一化
            nn.ReLU()  # 激活函数
        )
        # 定义自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lrc = LRCED(3)

    def fusion(self, f1, f2, f_vec):
        # 提取权重
        w1 = f_vec[:, 0, :, :].unsqueeze(1)  # 提取第一个通道的权重
        w2 = f_vec[:, 1, :, :].unsqueeze(1)  # 提取第二个通道的权重

        # 计算加权和
        out1 = (w1 * f1) + (w2 * f2)
        # 计算加权乘积
        out2 = (w1 * f1) * (w2 * f2)

        # 返回融合结果
        return out1 + out2
    def forward(self, x1, x2, x3, x4):
        In1 = self.wcm(x1, x2)
        In2 = self.wcm(x3, x4)
        f = torch.cat([In1, In2], dim =1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        out = self.fusion(In1, In2, f)
        return out
if __name__ == '__main__':
    # 创建RGB和深度输入的假设张量
    rgb_input = torch.randn(1, 3, 32, 32)  # RGB输入
    depth_input = torch.randn(1, 3, 32, 32)  # 深度输入
    x3 = torch.randn(1, 3, 32, 32)
    x4 = torch.randn(1, 3, 32, 32)
    # 通过WCMF模型
    wcm = many_WCMF(3)
    output = wcm(rgb_input, depth_input, x3, x4)

    # 打印输入和输出的shape
    print("RGB:", rgb_input.shape)
    print("深度:", depth_input.shape)
    print("输出形状:", output.shape)


