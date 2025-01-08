import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm.modules.mamba_simple import Mamba

class AddGatedNoise(nn.Module):
    def __init__(self):
        super(AddGatedNoise, self).__init__()

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * 0.1  # Adjust noise level as needed
            return x + noise
        return x


class Conv1DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(Conv1DTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                                 output_padding=output_padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class GLUConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GLUConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, 2 * out_channels, kernel_size, stride=stride, padding=padding)
        # self.conv = nn.utils.spectral_norm(self.conv)
    def forward(self, x):
        out = self.conv(x)
        out, gate = out.chunk(2, dim=1)
        return out * torch.sigmoid(gate)
ks=13
class RM_UNet(nn.Module):
    def __init__(self, dropout=0.1, activation="relu"):
        super(RM_UNet, self).__init__()
        self.conv1 = GLUConv1d(1, 16, ks, stride=2, padding=6)

        self.conv2 = GLUConv1d(16, 32, ks, stride=2, padding=6)

        self.conv3 = GLUConv1d(32, 64, ks, stride=2, padding=6)
        self.conv_transpose1 = Conv1DTranspose(64, 64, ks, stride=1, padding=6)
        self.conv_transpose2 = Conv1DTranspose(64, 32, ks, stride=2, padding=6, output_padding=1)
        self.conv_transpose3 = Conv1DTranspose(32, 16, ks, stride=2, padding=6, output_padding=1)
        self.conv_transpose4 = Conv1DTranspose(16, 1, ks, stride=2, padding=6, output_padding=1)
        self.elu = nn.ELU()
        # self.elu = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm1d(16)  ##原始BatchNorm1d,InstanceNorm1d
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.batch_norm5 = nn.BatchNorm1d(32)
        self.batch_norm6 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.man = Mamba(
            d_model=64,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=3,  # Local convolution width
            expand=2,  # Block expansion factor)
        )
        self.man2 = Mamba(
            d_model=64,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Conv + noise + sigmoid + Multiply for first block
        # x = x.permute(0, 2, 1)
        x0 = self.conv1(x)
        x0 = self.batch_norm1(x0)
        x1 = self.conv2(x0)
        x1 = self.batch_norm2(x1)
        x2 = self.conv3(x1)
        x2 = self.batch_norm3(x2)


        # Positional Encoding
        # x2_ = x2.permute(0, 2, 1)  
        x3 = self.man(x2)
        x4 = x3

        x5 = self.conv_transpose1(x4)  ##U net
        x5 = self.elu(x5)
        x5 = x5 + x2
        x5 = self.batch_norm4(x5)

        x6 = self.conv_transpose2(x5)
        x6 = self.elu(x6)
        # print(x6.size())
        x6 = x6 + x1
        x6 = self.batch_norm5(x6)

        x7 = self.conv_transpose3(x6)
        x7 = self.elu(x7)
        x7 = x7 + x0
        x8 = self.batch_norm6(x7)

        predictions = self.conv_transpose4(x8)
        return predictions


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1)


class TempAttention(nn.Module):
    def __init__(self):
        super(TempAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        # print('***')
        # print(out.size())
        out = self.conv(out)
        return torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.temp_attention = TempAttention()

    def forward(self, x):
        # print('@@@')
        # print(self.channel_attention(x).size())
        x = x * self.channel_attention(x)
        # print('###')
        # print(x.size())
        x = x * self.spatial_attention(x)
        # print('$$$')
        # print(x.size())
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, activation='leaky_relu'):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, padding=(kernel_size-1)//2)
        # self.batchnorm = nn.BatchNorm1d(out_channel, affine=True)  # Instance Normalization
        self.activation = getattr(F, activation)
        self.attention = CBAM(out_channel)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        output = self.conv(x)
        # output = self.batchnorm(output)
        output = self.activation(output)
        output = self.attention(output)
        output = self.maxpool(output)
        # print('$$$')
        # print(output.size())
        return output

class AttentionDeconvECA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, activation='leaky_relu', strides=2):
        super(AttentionDeconvECA, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride=strides,
                                         padding=(kernel_size-2)//2)
        # self.batchnorm = nn.BatchNorm1d(out_channel, affine=True)  ##另加
        # Map activation string to actual activation function
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)  # Use LeakyReLU
        elif activation == 'relu':
            self.activation = nn.ReLU()  # Use ReLU
        elif activation == 'linear':
            self.activation = nn.Identity()  # Use Identity (no activation)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.attention = CBAM(out_channel)

    def forward(self, x):
        output = self.deconv(x)
        # output = self.batchnorm(output)  ##另加
        output = self.activation(output)
        # output = self.attention(output)
        # print('***')
        # print(output.size())
        return output



class MAUnet(nn.Module):
    def __init__(self):
        super(MAUnet, self).__init__()
        self.b1 = EncoderBlock(1, 16, kernel_size=13)
        # self.manb1 = Mamba(
        #     d_model=256,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.b2 = EncoderBlock(16, 32, kernel_size=7)
        # self.manb2 = Mamba(
        #     d_model=128,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.b3 = EncoderBlock(32, 64, kernel_size=7)
        # self.manb3 = Mamba(
        #     d_model=64,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.b4 = EncoderBlock(64, 64, kernel_size=7)
        self.manb4 = Mamba(
            d_model=32,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=3,  # Local convolution width
            expand=2,  # Block expansion factor)
        )
        self.b5 = EncoderBlock(64, 1, kernel_size=7)
        # self.manb5 = Mamba(
        #     d_model=16,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.d5 = AttentionDeconvECA(1, 64, kernel_size=8)
        # self.mand5 = Mamba(
        #     d_model=32,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.d4 = AttentionDeconvECA(64, 64, kernel_size=8)
        # self.mand4 = Mamba(
        #     d_model=64,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.d3 = AttentionDeconvECA(64, 32, kernel_size=8)
        # self.mand3 = Mamba(
        #     d_model=128,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.d2 = AttentionDeconvECA(32, 16, kernel_size=8)
        # self.mand2 = Mamba(
        #     d_model=256,  # Model dimension d_model
        #     d_state=64,  # SSM state expansion factor
        #     d_conv=3,  # Local convolution width
        #     expand=2,  # Block expansion factor)
        # )
        self.d1 = AttentionDeconvECA(16, 1, activation='linear', kernel_size=14)
        # self.dense = nn.Linear(signal_size, signal_size)

    def forward(self, x):
        enc1 = self.b1(x)
        # enc1 = self.manb1(enc1)
        enc2 = self.b2(enc1)
        # enc2 = self.manb2(enc2)
        enc3 = self.b3(enc2)
        # enc3 = self.manb3(enc3)
        enc4 = self.b4(enc3)
        # enc4 = self.manb4(enc4)
        enc5 = self.b5(enc4)
        enc5 = self.manb5(enc5)
        dec5 = self.d5(enc5)
        # dec5 = self.mand5(dec5)
        dec4 = self.d4(dec5 + enc4)
        # dec4 = self.mand4(dec4)
        # dec4 = self.d4(enc4)
        dec3 = self.d3(dec4 + enc3)
        # dec3 = self.mand3(dec3)
        dec2 = self.d2(dec3 + enc2)
        # dec2 = self.mand2(dec2)
        dec1 = self.d1(dec2 + enc1)
        return dec1
