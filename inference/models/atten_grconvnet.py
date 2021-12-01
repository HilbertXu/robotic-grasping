import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock, TransformerEncoderLayer, NestedTensor, PositionEmbeddingSine


class AttentiveGenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(AttentiveGenerativeResnet, self).__init__()

        self.channel_size = channel_size

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.conv4 = nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 8)

        self.pe = PositionEmbeddingSine(num_pos_feats = channel_size * 4, normalize=True)
        self.attn_encode_layer1 = TransformerEncoderLayer(d_model=channel_size * 8, nhead=4, dim_feedforward=channel_size * 8 * 2, dropout=0.1)
        self.attn_encode_layer2 = TransformerEncoderLayer(d_model=channel_size * 8, nhead=4, dim_feedforward=channel_size * 8 * 2, dropout=0.1)

        # self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        # self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.conv5 = nn.ConvTranspose2d(channel_size * 8, channel_size * 4, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size * 4)

        self.conv6 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.bn6 = nn.BatchNorm2d(channel_size * 2)

        self.conv7 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.bn7 = nn.BatchNorm2d(channel_size)

        self.conv8 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, padding=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        # 3 x Conv
        # print("input: ", x_in.shape)
        x = F.relu(self.bn1(self.conv1(x_in))) # [bs, 4, 224, 224] -> [bs, 32, 224, 224]
        # print("conv1: ", x.shape)
        x = F.relu(self.bn2(self.conv2(x))) # [bs, 32, 224, 224] -> [bs, 64, 112, 112]
        # print("conv2: ", x.shape)
        x = F.relu(self.bn3(self.conv3(x))) # [bs, 64, 112, 112] -> [bs, 128, 56, 56]
        # print("conv3: ", x.shape)
        x = F.relu(self.bn4(self.conv4(x))) # [bs, 128, 56, 56] -> [bs, 256, 28, 28]
        # print("conv4: ", x.shape)

        bs, cs, fs = x.shape[:3]

        mask = torch.ones(bs, fs, fs).to(x.device)
        pos = self.pe(x, mask)
        pos = pos.reshape(bs, cs, -1).permute(0,2,1)

        x = x.reshape(bs, cs, -1).permute(0,2,1) # [bs, 256, 28, 28] -> [bs, 256, 784] -> [bs, 784, 256]
        # print(x.shape)

        x = self.attn_encode_layer1(x, pos)
        x = self.attn_encode_layer2(x, pos)

        x = x.reshape(bs, fs, fs, -1).permute(0,3,1,2) # [bs, 784, 256] -> [bs, 256, 784] -> [bs, 256, 28, 28]
        # print(x.shape)

        # 3 x ConvTranspose
        x = F.relu(self.bn5(self.conv5(x)))
        # print("deconv5: ", x.shape)
        x = F.relu(self.bn6(self.conv6(x)))
        # print("deconv6: ", x.shape)
        x = F.relu(self.bn7(self.conv7(x)))
        # print("deconv7: ", x.shape)
        x = self.conv8(x)
        # print("deconv8: ", x.shape)

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


if __name__ == "__main__":
    import torch

    fake_inp = torch.randn(2,4,224,224).float()

    print(fake_inp.shape)


    net = AttentiveGenerativeResnet()

    oup = net(fake_inp)
