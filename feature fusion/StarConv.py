import torch
import torch.nn as nn
from timm.models.layers import DropPath

class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)  
        self.bn = nn.BatchNorm2d(c2) 
        self.act = nn.SiLU()
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.bn(self.conv(x))              
        # return self.act(self.bn(self.conv(x)))  

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, g=dim)                   

        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)          # 用1x1卷积来代替全连接层。
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)          

        self.g = ConvBN(mlp_ratio * dim, dim, 1)  # Star
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim) 

        self.act = nn.ReLU6()                                              

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

    def forward(self, x):
        input = x  

        x = self.dwconv(x)  
        x1, x2 = self.f1(x), self.f2(x)  

        x = self.act(x1) * x2  

        x = self.g(x)  # starConv
        x = self.dwconv2(x)

        x = input + self.drop_path(x)  
        return x


if __name__ == '__main__':

    block = Star_Block(dim=32)  

    input = torch.rand(1, 32, 64, 64)  

    output = block(input) 
    print("input.shape:", input.shape)  
    print("output.shape:", output.shape)  

