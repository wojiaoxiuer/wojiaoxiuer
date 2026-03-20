import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # 从timm库中导入一些常用的层函数
from StarConv import Star_Block
act_layer = nn.ReLU  # 定义激活函数为ReLU
ls_init_value = 1e-6  # 定义初始化参数值
class LRCED(nn.Module):  # 定义LRCED模块类

    def __init__(self, dim, drop_path=0., dilation=3, **kwargs):
        super().__init__()  # 调用父类构造函数
        #self.star = Star_Block(dim)
        # 第一个深度可分离卷积序列，包含卷积、归一化和激活
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),  # 标准卷积
            nn.BatchNorm2d(dim),  # 批标准化
            act_layer())  # 激活函数

        # 第二个深度可分离卷积序列，包含扩张卷积、归一化和激活
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),  # 扩张卷积
            nn.BatchNorm2d(dim),  # 批标准化
            act_layer())  # 激活函数

        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 第一个逐点卷积
        self.act = act_layer()  # 激活函数
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 第二个逐点卷积

        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None  # 可学习的缩放参数
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 随机丢弃路径
        #self.star = Star_Block(dim)
    def forward(self, x):
        input = x  # 保存输入以供残差连接
        x = self.dwconv1(x) + x  # 应用第一个深度可分离卷积并添加残差
        # x = self.star(x)#星操作

        x = self.dwconv2(x) + x  # 应用第二个深度可分离卷积并添加残差

        # 因为全连接层默认对最后一个维度进行变化，因此需要调整维度
        x = x.permute(0, 2, 3, 1)  # 改变维度顺序 (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)  # 应用第一个逐点卷积
        x = self.act(x)      # 激活
        x = self.pwconv2(x)  # 应用第二个逐点卷积
        if self.gamma is not None:
            x = self.gamma * x  # 如果定义了gamma，则进行缩放
        x = x.permute(0, 3, 1, 2)  # 将维度顺序改回 (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)  # 添加残差连接，并应用随机丢弃路径
        return x  # 返回处理后的张量

if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)  # 创建随机输入张量
    model = LRCED(64)  # 实例化CED模型
    output = model(input)  # 前向传播
    print('input_size:', input.size())  # 打印输入尺寸
    print('output_size:', output.size())  # 打印输出尺寸
    print(model)

