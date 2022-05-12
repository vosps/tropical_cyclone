"""
https://towardsdatascience.com/u-net-b229b32b4a71
https://wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA

https://www.essoar.org/pdfjs/10.1002/essoar.10507812.1
"""
import torch
import torchvision
import numpy as np
# from fastai.data.core import DataLoaders
# from fastai.vision import *
# from fastai.metrics import error_rate, accuracy
import argparse
import logging
import sys
from pathlib import Path
# import torch
import torch.nn as nn
# import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
sns.set_style("white")

from unet_model import UNet

def train_net(net,
			  device,
			  epochs: int = 5,
			#   batch_size: int = 100,
			  batch_size: int = 1,
			  learning_rate: float = 0.0001,
			  val_percent: float = 0.1,
			  save_checkpoint: bool = True,
			  img_scale: float = 0.5,
			  amp: bool = False):
	# 1. Create dataset
	valid_X = torch.tensor(np.load('/user/work/al18709/tc_data_mswep/valid_X.npy')).unsqueeze(1)
	valid_y = torch.tensor(np.load('/user/work/al18709/tc_data_mswep/valid_y.npy')).unsqueeze(1)
	train_X = torch.tensor(np.load('/user/work/al18709/tc_data_mswep/train_X.npy')).unsqueeze(1)
	train_y = torch.tensor(np.load('/user/work/al18709/tc_data_mswep/train_y.npy')).unsqueeze(1)

	# 1.5 regrid to 256 x 256
	scale = nn.Upsample(size=(64,64), mode='nearest')
	# scale2 = nn.Upsample(size=(256,256), mode='nearest')
	scale2 = nn.Upsample(size=(128,128), mode='nearest')
	train_X = scale(train_X)
	train_y = scale2(train_y)
	valid_X = scale(valid_X)
	valid_y = scale2(valid_y)

	# 2. Split into train / validation partitions
	train_set = train_X
	val_set = valid_X
	n_train,_,_,_ = train_set.shape
	n_val,_,_,_ = val_set.shape
	batch_size = 100
	epochs = 100
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
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # goal: maximize Dice score
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

				assert images.shape[1] == net.n_channels, \
					f'Network has been defined with {net.n_channels} input channels, ' \
					f'but loaded images have {images.shape[1]} channels. Please check that ' \
					'the images are loaded correctly.'

				images = images.to(device=device, dtype=torch.float32)
				true_masks = true_masks.to(device=device, dtype=torch.float32)

				with torch.cuda.amp.autocast(enabled=amp):
					masks_pred = net(images)
					loss = criterion(masks_pred, true_masks)

					# save data
					y_pred = masks_pred[:,0,:,:].detach().cpu().numpy() #added the .cpu part
					y_true = true_masks[:,:,:].detach().cpu().numpy()
					X = images[:,0,:,:].cpu()
					np.save('data/y_pred.npy',y_pred)
					np.save('data/y_true.npy',y_true)
					np.save('data/X.npy',X)
					
					
					

				# optimizer.zero_grad(set_to_none=True) # 1
				optimizer.zero_grad() # 2
				# grad_scaler.scale(loss).backward() # 1
				# grad_scaler.step(optimizer) # 1
				loss.backward() # 2
				optimizer.step() # 2 update the weights

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
				#   epochs=25,
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



