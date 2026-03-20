import torch.nn as nn
import torch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == '__main__':
    block = SELayer(32)
    # 创建随机输入张量
    input = torch.rand(1, 32, 64, 64)
    # 通过网络传递输入张量
    output = block(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)