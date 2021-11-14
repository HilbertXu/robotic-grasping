from .unet_parts import *

from inference.models.grasp_model import GraspModel, ResidualBlock


class GR_UNet(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, bilinear=True):
        super(GR_UNet, self).__init__()
        self.n_channels = input_channels
        self.bilinear = True

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.pos_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.cos_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.sin_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.width_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(prob)
        self.dropout_cos = nn.Dropout(prob)
        self.dropout_sin = nn.Dropout(prob)
        self.dropout_wid = nn.Dropout(prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        # print("x1: ", x1.shape)
        x2 = self.down1(x1)
        # print("x2: ", x2.shape)
        x3 = self.down2(x2)
        # print("x3: ", x3.shape)
        x4 = self.down3(x3)
        # print("x4: ", x4.shape)
        x5 = self.down4(x4)
        # print("x5: ", x5.shape)
        x = self.up1(x5, x4)
        # print("up1: ", x.shape)
        x = self.up2(x, x3)
        # print("up2: ", x.shape)
        x = self.up3(x, x2)
        # print("up3: ", x.shape)
        x = self.up4(x, x1)
        # print("up4: ", x.shape)

        # Prediction head
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
