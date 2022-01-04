import noise
import plots


def train_model(*,
                model=None,
                mode=None,
                batch_gen_train=None,
                batch_gen_valid=None,
                noise_channels=None,
                latent_variables=None,
                epoch=None,
                steps_per_epoch=None,
                plot_samples=8,
                plot_fn=None):

    # for cond, _, _ in batch_gen_train.take(1).as_numpy_iterator(): #TODO: fix issue here
    # print(batch_gen_train.take(1).as_numpy_iterator())
    for cond, _, in batch_gen_train.take(1).as_numpy_iterator(): #TODO: remove one argument? no, needs to have 3 arguments, figure out what constant is
        print('looping through cond...')
        # img_shape = cond.shape[1:-1]
        img_shape = cond.shape[1:] # so far this is the only way to get the correct image shape
        batch_size = cond.shape[0]
    del cond
    print(img_shape)
    print(batch_size)
    if mode == 'GAN':
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_epoch, training_ratio=5)

    elif mode == 'VAEGAN':
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_epoch, training_ratio=5)

    elif mode == 'det':
        loss_log = model.train(batch_gen_train, steps_per_epoch)

    plots.plot_sequences(model.gen,
                         mode,
                         batch_gen_valid,
                         epoch,
                         noise_channels=noise_channels,
                         latent_variables=latent_variables,
                         num_samples=plot_samples,
                         out_fn=plot_fn)

    return loss_log
