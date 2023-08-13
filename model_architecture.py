import torch
from torch import nn


class Down(nn.Module):
    """convolution => [BN] => LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """transpose_conv => [BN] => LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)


class Anomaly(nn.Module):
    def __init__(
            self,
            n_channels=3,
            middle_channels=(32, 64, 128)
    ):
        super(Anomaly, self).__init__()
        self.n_channels = n_channels
        self.middle_channels_down = middle_channels
        self.middle_channels_up = tuple(reversed(middle_channels))

        self.initial = Down(self.n_channels, self.middle_channels_down[0])

        self.downs = nn.ModuleList([
                Down(
                    self.middle_channels_down[i], self.middle_channels_down[i + 1]
                ) for i in range(len(self.middle_channels_down) - 1)
            ])

        self.ups = nn.ModuleList([
                Up(
                    self.middle_channels_up[i], self.middle_channels_up[i + 1]
                ) for i in range(len(self.middle_channels_up) - 1)
            ])

        self.out = nn.Sequential(
            nn.ConvTranspose2d(self.middle_channels_up[-1], self.n_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial(x)
        for down in self.downs:
            x = down(x)
        for up in self.ups:
            x = up(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    model = Anomaly()
    print('Num params: ', sum(p.numel() for p in model.parameters()))
    test_x = torch.rand(1, 3, 64, 64)
    test_y = model(test_x)
    print('Output shape: ', test_y.shape)
