from collections import OrderedDict

import megengine as mge
import megengine.module as M
import megengine.functional as F


def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = M.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = M.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = M.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = M.LeakyReLU(negative_slope=0.125)

    return M.Sequential(modules)


class EncoderBlock(M.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=True, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True, has_relu=False)

        self.proj = (
            M.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True, has_relu=False)
        )
        self.relu = M.LeakyReLU(negative_slope=0.125)

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
            )
        )

    return M.Sequential(*blocks)


class DecoderBlock(M.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=False,
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(M.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = M.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True)

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class Network(M.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True) # 16*256*256
        self.enc1 = EncoderStage(in_channels=8, out_channels=32, num_blocks=3)
        self.enc2 = EncoderStage(in_channels=32, out_channels=64, num_blocks=4)
        self.enc3 = EncoderStage(in_channels=64, out_channels=128, num_blocks=4)
        # self.enc4 = EncoderStage(in_channels=64, out_channels=128, num_blocks=4)
        self.encdec = Conv2D(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        # self.dec1 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec2 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=32, out_channels=16)
        self.dec4 = DecoderStage(in_channels=16, skip_in_channels=8, out_channels=16)
        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
    def forward(self, inp):
        inp=inp.reshape(-1,1,256,256)
        conv0 = self.conv0(inp) #4*256*256  #16*512*512
        conv1 = self.enc1(conv0) #16*128*128  #64*256*256
        conv2 = self.enc2(conv1) #32*64*64  #128*128*128
        conv3 = self.enc3(conv2) #64*32*32  #256*64*64
        # conv4 = self.enc4(conv3) #128*16*16 #512*32*32

        conv5 = self.encdec(conv3) #32*16*16 #64*32*32

        # up3 = self.dec1((conv5, conv3)) #32*32*32 #64*64*64
        up2 = self.dec2((conv5, conv2)) #16*64*64 #32*128*128
        up1 = self.dec3((up2, conv1)) #16*128*128 #32*256*256
        x = self.dec4((up1, conv0)) #4*256*256  #16*512*512

        x = self.out0(x)
        x = self.out1(x)

        pred = inp + x
        pred = pred.reshape(-1,256, 256)
        return pred