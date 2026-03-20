import torch
import torch.nn as nn
from LRCED import LRCED

class WCMF(nn.Module):
    def __init__(self, channel=256):
        super(WCMF, self).__init__()
        
        self.conv_r1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0), 
            nn.BatchNorm2d(channel),              
            nn.ReLU()                             
        )

        self.conv_d1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),  
            nn.BatchNorm2d(channel),             
            nn.ReLU()                             
        )
        
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2*channel, channel, 3, 1, 1), 
            nn.BatchNorm2d(channel),                 
            nn.ReLU()                                
        )
        
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, 2, 3, 1, 1),         
            nn.BatchNorm2d(2),                      
            nn.ReLU()                               
        )
       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lrc = LRCED(3)

    def fusion(self, f1, f2,  f_vec):
       
        w1 = f_vec[:, 0, :, :].unsqueeze(1) 
        w2 = f_vec[:, 1, :, :].unsqueeze(1)  
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2
        
    def forward(self, rgb, depth):
        rgb = self.lrc(rgb)
        depth = self.lrc(depth)
        Fr = self.conv_r1(rgb) #+ self.lrc(rgb)
        Fd = self.conv_d1(depth) #+ self.lrc(depth)
        f = torch.cat([Fr, Fd], dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo
class many_WCMF(nn.Module):
    def __init__(self, channel=256):
        super(many_WCMF, self).__init__()
        self.wcm = WCMF(3)
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, 3, 1, 1),  
            nn.BatchNorm2d(channel),  
            nn.ReLU()  
        )
       
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, 2, 3, 1, 1),  
            nn.BatchNorm2d(2),  
            nn.ReLU()  
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lrc = LRCED(3)

    def fusion(self, f1, f2, f_vec):
        
        w1 = f_vec[:, 0, :, :].unsqueeze(1)  
        w2 = f_vec[:, 1, :, :].unsqueeze(1)  
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
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
    
    rgb_input = torch.randn(1, 3, 32, 32)  
    depth_input = torch.randn(1, 3, 32, 32)  
    x3 = torch.randn(1, 3, 32, 32)
    x4 = torch.randn(1, 3, 32, 32)
    
    wcm = many_WCMF(3)
    output = wcm(rgb_input, depth_input, x3, x4)

    print("RGB:", rgb_input.shape)
    print("深度:", depth_input.shape)
    print("输出形状:", output.shape)


