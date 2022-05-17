import torch
import numpy as np
import torch.nn as nn
from unet_new import UNet

# load validation data
valid_X = torch.tensor(np.load('/user/work/al18709/tc_data_mswep/valid_X.npy')).unsqueeze(1)

# define device and assign things to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss().to(device)
net = UNet(n_channels=1, n_classes=1, bilinear=True)
net.load_state_dict(torch.load('/user/home/al18709/work/cnn/unet.pth', map_location=device))
net.eval()
net.to(device=device)
images = valid_X.to(device=device, dtype=torch.float32)

# make predictions
print(images.shape)
with torch.no_grad():
	masks_pred_1 = net(images[0:8000])
# torch.cuda.empty_cache()
	masks_pred_2 = net(images[8001:16253])
pred_1 = masks_pred_1[:,0,:,:].detach().cpu().numpy() #added the .cpu part
pred_2 = masks_pred_2[:,0,:,:].detach().cpu().numpy() #added the .cpu part
print(pred_1.shape)
print(pred_2.shape)
pred = np.append(pred_1,pred_2,axis=0)
print(pred.shape)
# X = images[:,0,:,:].cpu()
np.save('/user/home/al18709/work/cnn/unet_valid.npy',pred)



