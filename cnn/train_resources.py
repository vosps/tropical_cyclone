"""
https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5
https://arxiv.org/abs/1505.04597
https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical
https://debuggercafe.com/image-super-resolution-using-deep-learning-and-pytorch/
https://docs.fast.ai/tutorial.siamese.html 
https://course18.fast.ai/ml.html
https://course17.fast.ai/lessons/lesson7.html

https://github.com/milesial/Pytorch-UNet/blob/master/train.py
https://cedrickchee.gitbook.io/knowledge/courses/fast.ai/deep-learning-part-2-cutting-edge-deep-learning-for-coders/2018-edition/lesson-14-image-segmentation
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py

"""

import torch
import numpy as np
from fastai.data.core import DataLoaders
from fastai.vision import *
from fastai.metrics import error_rate, accuracy


valid_X = torch.tensor(np.load('/OLD/work/al18709/tc_data/cv_X.npy')).unsqueeze(1)
valid_y = torch.tensor(np.load('/OLD/work/al18709/tc_data/cv_y.npy'))
train_X = torch.tensor(np.load('/OLD/work/al18709/tc_data/train_X.npy')).unsqueeze(1)
train_y = torch.tensor(np.load('/OLD/work/al18709/tc_data/train_y.npy'))



import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import TensorDataset
# from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet import UNet


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
			self.up = nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True)
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
		self.prelu = nn.PReLU()
		self.up = nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		x = self.up(x)
		# TODO: should this just be upscaled or should I try to use pixel shuffle here?
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		
		factor = 2 if bilinear else 1
		self.down3 = Down(256, 512 // factor)
		# self.down4 = Down(512, 1024 // factor)
		# self.up1 = Up(1024, 512 // factor, bilinear)
		self.up1 = Up(512, 256 // factor, bilinear)
		self.up2 = Up(256, 128 // factor, bilinear)
		self.up3 = Up(128, 64, bilinear)
		# self.up4 = Up(64, 32, bilinear)
		self.outc = OutConv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		# print('x1 ',x1.shape)
		x2 = self.down1(x1)
		# print('x2 ',x2.shape)
		x3 = self.down2(x2)
		# print('x3 ',x3.shape)
		x4 = self.down3(x3)
		# print('x4 ',x4.shape)
		# x5 = self.down4(x4)
		# print('x5 ',x5.shape)
		x = self.up1(x4, x3)
		# print('x ',x.shape)
		x = self.up2(x, x2)
		# print('x ',x.shape)
		x = self.up3(x, x1)
		# print('x ',x.shape)
		# x = self.up4(x, x1)
		# print('x ',x.shape)
		logits = self.outc(x)
		# print('logits',logits.shape)

		return logits
		
		# TODO: figure out how to get x to correct shape - have a look at upscalign architecture

		return logits

def train_net(net,
			  device,
			  epochs: int = 5,
			#   batch_size: int = 100,
			  batch_size: int = 1,
			  learning_rate: float = 0.001,
			  val_percent: float = 0.1,
			  save_checkpoint: bool = True,
			  img_scale: float = 0.5,
			  amp: bool = False):
	# 1. Create dataset


	# # 2. Split into train / validation partitions
	train_set = train_X
	val_set = valid_X
	n_train,_,_,_ = train_set.shape
	n_val,_,_,_ = val_set.shape
	batch_size=100
	# batch_size=1

	# 3. Create data loaders
	loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
	train = TensorDataset(train_X, train_y)
	train_loader = DataLoader(train, **loader_args)
	val = TensorDataset(valid_X, valid_y)
	val_loader = DataLoader(val, **loader_args)

	# (Initialize logging)
	experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
	experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
								  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
								  amp=amp))

	logging.info(f'''Starting training:
		Epochs:		  {epochs}
		Batch size:	  {batch_size}
		Learning rate:   {learning_rate}
		Training size:   {n_train}
		Validation size: {n_val}
		Checkpoints:	 {save_checkpoint}
		Device:		  {device.type}
		Images scaling:  {img_scale}
		Mixed Precision: {amp}
	''')

	# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	# optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
	optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=learning_rate)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
	grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
	# criterion = nn.CrossEntropyLoss()
	criterion = nn.MSELoss().to(device)
	global_step = 0

	# 5. Begin training
	for epoch in range(epochs):
		net.train()
		epoch_loss = 0
		with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
			for batch in train_loader:
				images = batch[0]
				true_masks = batch[1]
				# print('true mask shape: ',true_masks.shape)
				# true_masks = true_masks.unsqueeze(1)
				# print('true mask shape: ',true_masks.shape)


				assert images.shape[1] == net.n_channels, \
					f'Network has been defined with {net.n_channels} input channels, ' \
					f'but loaded images have {images.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				images = images.to(device=device, dtype=torch.float32)
				true_masks = true_masks.to(device=device, dtype=torch.float32)

				with torch.cuda.amp.autocast(enabled=amp):
					masks_pred = net(images)
					# loss = criterion(masks_pred, true_masks) \
					# 	   + dice_loss(F.softmax(masks_pred, dim=1).float(),
					# 				   F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
					# 				   multiclass=True)
					loss = criterion(masks_pred, true_masks)
					

				# optimizer.zero_grad(set_to_none=True)
				# grad_scaler.scale(loss).backward()
				# grad_scaler.step(optimizer)
				# grad_scaler.update()
				optimizer.zero_grad()
				loss.backward()

				pbar.update(images.shape[0])
				global_step += 1
				epoch_loss += loss.item()
				experiment.log({
					'train loss': loss.item(),
					'step': global_step,
					'epoch': epoch
				})
				pbar.set_postfix(**{'loss (batch)': loss.item()})

				# Evaluation round
				if global_step % (n_train // (10 * batch_size)) == 0:
					histograms = {}
					for tag, value in net.named_parameters():
						tag = tag.replace('/', '.')
						histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
						# histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

					# val_score = evaluate(net, val_loader, device) TODO: create own evaulate function
					# scheduler.step(val_score)

					# logging.info('Validation Dice score: {}'.format(val_score))
					# experiment.log({
					# 	'learning rate': optimizer.param_groups[0]['lr'],
					# 	'validation Dice': val_score,
					# 	'images': wandb.Image(images[0].cpu()),
					# 	'masks': {
					# 		'true': wandb.Image(true_masks[0].float().cpu()),
					# 		'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
					# 	},
					# 	'step': global_step,
					# 	'epoch': epoch,
					# 	**histograms
					# })

		# if save_checkpoint:
		# 	Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
		# 	torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
		# 	logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
	parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
	parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
	parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
	parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
						help='Learning rate', dest='lr')
	parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
	parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
	parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
						help='Percent of the data that is used as validation (0-100)')
	parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

	# Change here to adapt to your data
	# n_channels=3 for RGB images
	# n_classes is the number of probabilities you want to get per pixel
	net = UNet(n_channels=1, n_classes=1, bilinear=True)

	logging.info(f'Network:\n'
				 f'\t{net.n_channels} input channels\n'
				 f'\t{net.n_classes} output channels (classes)\n'
				 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

	if args.load:
		net.load_state_dict(torch.load(args.load, map_location=device))
		logging.info(f'Model loaded from {args.load}')

	net.to(device=device)
	try:
		train_net(net=net,
				  epochs=args.epochs,
				  batch_size=args.batch_size,
				  learning_rate=args.lr,
				  device=device,
				  img_scale=args.scale,
				  val_percent=args.val / 100,
				  amp=args.amp)
	except KeyboardInterrupt:
		torch.save(net.state_dict(), 'INTERRUPTED.pth')
		logging.info('Saved interrupt')
		sys.exit(0)








exit()

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:	# print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')