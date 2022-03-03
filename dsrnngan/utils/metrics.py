# import numpy
import numpy as np
# from numpy import cov
# from numpy import trace
# from numpy import iscomplexobj
# from numpy import asarray
# from numpy.random import randint
# from scipy.linalg import sqrtm
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
# from keras.datasets.mnist import load_data
# from skimage.transform import resize
 
# scale an array of images to a new size
# def scale_images(images, new_shape):
# 	images_list = list()
# 	for image in images:
# 		# resize with nearest neighbor interpolation
# 		new_image = resize(image, new_shape, 0)
# 		# store
# 		images_list.append(new_image)
# 	return asarray(images_list)
 

# https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# https://live.ece.utexas.edu/research/Quality/index_algorithms.htm 
# https://faculty.ucmerced.edu/mhyang/papers/eccv14_super.pdf
# VIF https://github.com/pavancm/Visual-Information-Fidelity---Python



def calculate_crps(observation, forecasts):
	# forecasts = forecasts[...,None]
	fc = forecasts.copy()
	fc.sort(axis=-1)
	obs = observation
	fc_below = fc < obs[..., None]
	crps = np.zeros_like(obs)

	for i in range(fc.shape[-1]):
		below = fc_below[..., i]
		weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
		crps[below] += weight * (obs[below]-fc[..., i][below])

	for i in range(fc.shape[-1] - 1, -1, -1):
		above = ~fc_below[..., i]
		k = fc.shape[-1] - 1 - i
		weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
		crps[above] += weight * (fc[..., i][above] - obs[above])

	return crps


# calculate frechet inception distance
def calculate_fid(images1, images2):

	# prepare the inception v3 model
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
	# model = InceptionV3(include_top=False, pooling='avg', input_shape=(100,100))

	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid



# # define two fake collections of images
# images1 = randint(0, 255, 10*32*32*3)
# images1 = images1.reshape((10,32,32,3))
# images2 = randint(0, 255, 10*32*32*3)
# images2 = images2.reshape((10,32,32,3))
# print('Prepared', images1.shape, images2.shape)
# # convert integer to floating point values
# images1 = images1.astype('float32')
# images2 = images2.astype('float32')
# # resize images
# images1 = scale_images(images1, (299,299,3))
# images2 = scale_images(images2, (299,299,3))
# print('Scaled', images1.shape, images2.shape)
# # pre-process images
# images1 = preprocess_input(images1)
# images2 = preprocess_input(images2)

