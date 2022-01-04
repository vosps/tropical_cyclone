""" Parts of the U-Net model 

https://www.youtube.com/watch?v=YTIMERuz4jg

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		# print('in channels: ',in_channels)
		# print('mid channels: ', mid_channels)
		# print('out channels: ',out_channels)
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), # kernel size is size of convolving kernel
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
		
	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			# self.up = nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True)
			self.up = nn.Upsample(size = 2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
			# would a Transpose Convolutional layer be good here?
		else:
			self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=10)
		self.prelu = nn.PReLU() #change to relu and see what happens
		# self.relu = nn.ReLU() # do relu to stop the negative? Seems to be more blurrier and still negative - though this was with 5 epochs
		# self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # TODO: change to scale factor 2 maybe?
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		x = self.up(x)
		# TODO: should this just be upscaled or should I try to use pixel shuffle here?
		return self.conv(x)