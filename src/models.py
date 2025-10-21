import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()
        self.d1 = self.down(in_channels, base)
        self.d2 = self.down(base, base*2)
        self.d3 = self.down(base*2, base*4)
        self.d4 = self.down(base*4, base*8)
        self.d5 = self.down(base*8, base*8)
        self.d6 = self.down(base*8, base*8)
        self.u1 = self.up(base*8, base*8)
        self.u2 = self.up(base*16, base*8)
        self.u3 = self.up(base*16, base*4)
        self.u4 = self.up(base*8,  base*2)
        self.u5 = self.up(base*4,  base)
        self.u6 = nn.ConvTranspose2d(base*2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def down(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def up(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2)
        d4 = self.d4(d3); d5 = self.d5(d4); d6 = self.d6(d5)
        u1 = self.u1(d6)
        u2 = self.u2(torch.cat([u1, d5], 1))
        u3 = self.u3(torch.cat([u2, d4], 1))
        u4 = self.u4(torch.cat([u3, d3], 1))
        u5 = self.u5(torch.cat([u4, d2], 1))
        out = self.u6(torch.cat([u5, d1], 1))
        return self.tanh(out)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64,128,256,512]):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features[0], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = features[0]
        for f in features[1:]:
            layers += [nn.Conv2d(ch, f, 4, 2, 1, bias=False), nn.BatchNorm2d(f), nn.LeakyReLU(0.2, inplace=True)]
            ch = f
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
