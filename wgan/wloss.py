from tensorflow.keras import backend as K

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)

# def wasserstein_loss(y_true, y_pred):
#     # weight the most important areas to penalise a high loss in those areas
#     # right now the zero values of rainfall are not penalised because they are just zero
#     # so we want to make sure that the loss where there is rainfall is small.
#     # method taken from patch gan in Recycling Discriminator: Towards Opinion-Unaware Image Quality Assessment Using Wasserstein GAN
#     # they conclude that wasserstein distance is less good when it just looks over the whole image
#     # they develop a method where aspects of their images are weighted higher based on whether they are more important
#     # for example, they have an edge detection thing where it helps them differentiate background and foreground and then they weight the foreground more
#     # I adapt this for rainfall where I weight the rainfall values higher than drizzle or low rain
#     # this forces the model to pay more attention to these values
#     # y_max = K.max(y_true) # gamma 
#     # image_relevance_weighting = y_true / y_max # Wsod / gamma to get values bwteen 0 and 1
#     # # weight_avg = K.ones((y_true.shape)) # Wavg is a matrix of ones because we don't want 0 weight on everything
#     # weight_avg = K.ones_like(y_true)
#     # weight = weight_avg + image_relevance_weighting
#     # sigma = K.sum(weight)
#     # wwloss = K.mean(y_true * y_pred, axis=-1) * weight * (1/sigma)

#     image_relevance_weighting = y_true # not botering with Wsod / gamma to get values bwteen 0 and 1, doesn't need to be between 0 and 1?
#     # weight_avg = K.ones((y_true.shape)) # Wavg is a matrix of ones because we don't want 0 weight on everything
#     weight_avg = K.ones_like(y_true)
#     weight_avg_2 = K.mean(y_true[:,:,:,0])
#     weight = weight_avg + image_relevance_weighting
#     # sigma = K.sum(weight)
#     wloss = K.mean(y_true * y_pred, axis=-1)
#     wwloss = wloss * weight + weight_avg_2

#     return wwloss