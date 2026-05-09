import torch  
import torch.nn as nn  
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  
from StarConv import Star_Block
act_layer = nn.ReLU  
ls_init_value = 1e-6  
class LRCED(nn.Module):  

    def __init__(self, dim, drop_path=0., dilation=3, **kwargs):
        super().__init__()  
        #self.star = Star_Block(dim)
       
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),  
            nn.BatchNorm2d(dim),  
            act_layer())  

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),  
            nn.BatchNorm2d(dim),  
            act_layer())  

        self.pwconv1 = nn.Linear(dim, 4 * dim)  
        self.act = act_layer()  
        self.pwconv2 = nn.Linear(4 * dim, dim)  

        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  
        self.star = Star_Block(dim)
    def forward(self, x):
        input = x  
        x = self.dwconv1(x) + x  
        x = self.star(x)
        x = self.dwconv2(x) + x  
        x = x.permute(0, 2, 3, 1)  
        x = self.pwconv1(x)  
        x = self.act(x)      
        x = self.pwconv2(x)  
        if self.gamma is not None:
            x = self.gamma * x  
        x = x.permute(0, 3, 1, 2)  

        x = input + self.drop_path(x)  
        return x 

if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)  
    model = LRCED(64)  
    output = model(input)  
    print('input_size:', input.size())  
    print('output_size:', output.size())  
    print(model)

