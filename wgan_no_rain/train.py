import noise
import plots


def train_model(*,
                model=None,
                mode=None,
                batch_gen_train=None,
                batch_gen_valid=None,
                noise_channels=None,
                latent_variables=None,
                checkpoint=None,
                steps_per_checkpoint=None,
                plot_samples=8,
                plot_fn=None):

    for cond, _, _ in batch_gen_train.take(1).as_numpy_iterator():
    # for cond, _ in batch_gen_train.take(1).as_numpy_iterator():
        # img_shape = cond.shape[1:-1]
        img_shape = cond.shape
        batch_size = cond.shape[0]
    del cond

    if mode == 'GAN':
        print('training gan')
        # noise_shape = (img_shape[0], img_shape[1], noise_channels)
        print('img_shape',img_shape)
        # noise_shape = (1, noise_channels)
        noise_shape = (10,10, noise_channels)
        noise_hr_shape = (50,50, noise_channels)
        # noise_shape = (5,5, noise_channels)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        noise_hr_gen = noise.NoiseGenerator(noise_hr_shape, batch_size=batch_size)
        print('noise shape: ', noise_shape)
        print('noise_hr shape: ', noise_hr_shape)
        print(batch_gen_train)
        loss_log = model.train(batch_gen_train, noise_gen, noise_hr_gen,
                               steps_per_checkpoint, training_ratio=6) #used to be 5

    elif mode == 'VAEGAN':
        print('training vaegan')
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_gen = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_gen,
                               steps_per_checkpoint, training_ratio=5)

    elif mode == 'det':
        loss_log = model.train(batch_gen_train, steps_per_checkpoint)

    plots.plot_sequences(model.gen,
                         mode,
                         batch_gen_valid,
                         checkpoint,
                         noise_channels=noise_channels,
                         latent_variables=latent_variables,
                         num_samples=plot_samples,
                         out_fn=plot_fn)

    return loss_log
