import numpy as np

# pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/validation_pred-opt_5_normal_problem.npy')

set = 'validation'
real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_improve.npy' % set)[:,:,:,0]
inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_improve.npy' % set)[:,:,:,0]
pred_cnn = np.load('/user/home/al18709/work/cnn/unet_valid_2.npy')
pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_5_normal_problem.npy' % set,mmap_mode='r')[:,:,:,0]
pred_vaegan = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_5_better_spread-error.npy' % set,mmap_mode='r')[:,:,:,0]
pred_gan_ensemble = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_5_normal_problem.npy' % set,mmap_mode='r')
pred_vaegan_ensemble = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_5_better_spread-error.npy' % set,mmap_mode='r')
print(pred_gan.shape)