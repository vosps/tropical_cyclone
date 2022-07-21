""" Full assembly of the parts to form the complete network. """

from unet_parts import *

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 128)
		self.down1 = Down(128, 256)
		factor = 2 if bilinear else 1
		self.down3 = Down(256, 512 // factor)
		self.up1 = Up(512, 256 // factor, bilinear,scale_factor=2)
		self.up2 = Up(256, 128 // factor, bilinear,scale_factor=2)
		self.up3 = UpUp(64,32, n_classes,scale_factor=5)
		# self.up4 = UpUp(32,16, n_classes,scale_factor=5)
		self.outc = OutConv(32, n_classes,scale_factor=2)

	def forward(self, x):

		x1 = self.inc(x)
		print('x1',x1.shape)
		x2 = self.down1(x1)
		print('x2',x2.shape)
		x3 = self.down3(x2)
		print('x3',x3.shape)

		x = self.up1(x3, x2)
		print('up1 x',x.shape)
		x = self.up2(x, x1)
		print('up 2 x',x.shape)
		x = self.up3(x)
		print('up 3 x',x.shape)
		# x = self.up4(x)
		# print('up 4 x',x.shape)
		logits = self.outc(x)
		print('logits',logits.shape)
		return logits
		
