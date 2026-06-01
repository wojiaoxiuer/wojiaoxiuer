import math
import torch
from torch import nn

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class FCAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))#根据log通道数加b来控制卷积核大小，通道数越大代表需要感知的局部范围越大，使用的k也越大
        k = t if t % 2 else t + 1#类似ECA代码自适应调节核大小，保证核为奇数才可以保证对称填充
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)#这里两个卷积的区别就在于1d卷积是使用一个1Xk大小的卷积核去卷U，因为是填充后的有效卷积亦称为same卷积(二者等效)每次卷会涉及到多个数字相乘再相加，可以等价于左乘一个带状矩阵(自己思考一下),然后2d的1x1卷积实际上就等于一个卷积核就一个数字对每个U中的每个元素各数乘一次，于是相当于左乘一个对角矩阵
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)Ul
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)Ug
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input*out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1,64,256,256)
    model = FCAttention(channel=64)
    output = model (input)
    print('input_size:', input.size())
    print('output_size:', output.size())
