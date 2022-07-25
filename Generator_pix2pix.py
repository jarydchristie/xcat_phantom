#!/usr/bin/env python
# coding: utf-8

# In[78]:


import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1,output_padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=32):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, features, 7, 1, 3, bias = False, padding_mode="reflect"),
                                          nn.ReLU(0.2),)#a
        self.down1 = Block(features, features * 2, down=True, act="relu", use_dropout=False) #a/2
        self.down2 = Block(features * 2, features * 3, down=True, act="relu", use_dropout=False)#a/4
        self.down3 = Block(features * 3, features * 4, down=True, act="relu", use_dropout=False)#a/8
        self.down4 = Block(features * 4, features * 6, down=True, act="relu", use_dropout=False)#a/16
        self.down5 = Block(features * 6, features * 8, down=True, act="relu", use_dropout=False)#a/32
        
        self.bottleneck1 = nn.Sequential(nn.Conv2d(features * 8, features * 8, kernel_size = 7, stride = 1, padding =3)
                                         , nn.ReLU())#a/32
        self.bottleneck2 = nn.Sequential(nn.Conv2d(features * 8, features * 8, kernel_size = 7, stride =1, padding=3)
                                         , nn.ReLU())#a/32
        self.bottleneck3 = nn.Sequential(nn.Conv2d(features * 8, features * 8, kernel_size = 7, stride = 1, padding =3)
                                         , nn.ReLU())#a/32

        self.up1 = Block(features * 8, features * 6, down=False, act="relu", use_dropout=False)#a/16
        self.up2 = Block(features * 6 * 2, features * 4, down=False, act="relu", use_dropout=False)#a/8
        self.up3 = Block(features * 4 * 2, features * 3, down=False, act="relu", use_dropout=False)#a/4
        self.up4 = Block(features * 3 * 2, features * 2, down=False, act="relu", use_dropout=False)#a/2
        self.up5 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)#a
        self.final_up = nn.Sequential(nn.ConvTranspose2d(features*2, in_channels, kernel_size=7, stride=1, padding=3),
                                      nn.Tanh(),)

    def forward(self, x):
        d1 = self.initial_down(x.double())
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        bottleneck1 = self.bottleneck1(d6)
        bottleneck2 = self.bottleneck2(bottleneck1)
        bottleneck3 = self.bottleneck3(bottleneck2)
        up1 = self.up1(bottleneck3)
        up2 = self.up2(torch.cat([up1, d5], 1))
        up3 = self.up3(torch.cat([up2, d4], 1))
        up4 = self.up4(torch.cat([up3, d3], 1))
        up5 = self.up5(torch.cat([up4, d2], 1))
        return self.final_up(torch.cat([up5, d1], 1))


#def test():
#    x = torch.randn((1, 3, 512, 512))
#    model = Generator(in_channels=3, features=32)
#    preds = model(x)
#    print(model)
#    print(preds.shape)


#if __name__ == "__main__":
#    test()


# In[ ]:




