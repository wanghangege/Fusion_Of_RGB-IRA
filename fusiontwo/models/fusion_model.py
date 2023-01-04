import torch
from torch import nn


class ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(ReflectConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = ReflectConv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = ReflectConv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = ReflectConv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = ReflectConv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = ReflectConv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = ReflectConv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = ReflectConv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = ReflectConv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = CMDAF(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))

        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))
        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = ReflectConv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = ReflectConv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = ReflectConv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = ReflectConv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class PIAFusion(nn.Module):
    def __init__(self):
        super(PIAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)
        return fused_image
