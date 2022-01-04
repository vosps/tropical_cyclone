import gc

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import generic_utils
from layers import GradientPenalty, RandomWeightedAverage
from meta import Nontrainable, input_shapes, ensure_list
from meta import save_opt_weights, load_opt_weights
from wloss import wasserstein_loss
from vaegantrain import VAE_trainer

class WGANGP(object):

    def __init__(
        self,
        gen,
        disc,
        mode,
        gradient_penalty_weight=10, 
        lr_disc=0.0001,
        lr_gen=0.0001,
        avg_seed=None,
        kl_weight=None
    ):

        self.gen = gen
        self.disc = disc
        self.mode = mode 
        self.gradient_penalty_weight = gradient_penalty_weight
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.kl_weight = kl_weight
        self.build_wgan_gp()

    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5"
        }
        return fn

    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])
        
        with Nontrainable(self.disc):
            self.gen_trainer._make_train_function()
            load_opt_weights(self.gen_trainer,
                load_files["gen_opt_weights"])
        with Nontrainable(self.gen):
            self.disc_trainer._make_train_function()
            load_opt_weights(self.disc_trainer,
                load_files["disc_opt_weights"])


    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])


    def build_wgan_gp(self):

        # find shapes for inputs
        if self.mode == 'GAN':
            cond_shapes = input_shapes(self.gen, "generator_input")
            # const_shapes = input_shapes(self.gen, "const_input") # just the shape of what would traditionally be a constant variable
            noise_shapes = input_shapes(self.gen, "noise_input")
        elif self.mode == 'VAEGAN':
            cond_shapes = input_shapes(self.gen.encoder, "generator_input")
            const_shapes = input_shapes(self.gen.encoder, "const_input")
            noise_shapes = input_shapes(self.gen.decoder, "noise_input")
        sample_shapes = input_shapes(self.disc, "generator_output")
        
        # Create generator training network
        with Nontrainable(self.disc):
            if self.mode == 'GAN':
                # create list of input shapes for each type of input
                cond_in = [Input(shape=s) for s in cond_shapes]
                # const_in = [Input(shape=s) for s in const_shapes]
                noise_in = [Input(shape=s) for s in noise_shapes]
                # gen_in = cond_in + const_in + noise_in
                gen_in = cond_in + noise_in
                if self.mode == 'GAN':
                    gen_out = self.gen(gen_in)
                elif self.mode == 'VAEGAN':
                    encoder_in = cond_in + const_in
                    (encoder_mean, encoder_log_var) = self.gen.encoder(encoder_in)
                    decoder_in = [encoder_mean,encoder_log_var,noise_in,const_in]
                    gen_out = self.gen.decoder(decoder_in)
                gen_out = ensure_list(gen_out)
                print('gen_in',gen_in)
                print('gen_out', gen_out)
                # disc_in_gen = cond_in + const_in + [gen_out]
                disc_in_gen = cond_in + [gen_out]
                disc_out_gen = self.disc(disc_in_gen) #TODO: remove const somerwhere to get this to work!
                self.gen_trainer = Model(inputs=gen_in, 
                                         outputs=disc_out_gen, 
                                         name='gen_trainer')
            elif self.mode == 'VAEGAN':
                self.gen_trainer = VAE_trainer(self.gen, self.disc, self.kl_weight)

        # Create discriminator training network
        # for now remove const
        with Nontrainable(self.gen):
            cond_in = [Input(shape=s,name='generator_input') for s in cond_shapes]
            # const_in = [Input(shape=s,name='constants') for s in const_shapes]
            noise_in = [Input(shape=s,name='noise_input') for s in noise_shapes]
            sample_in = [Input(shape=s,name='generator_output') for s in sample_shapes]
            # gen_in = cond_in + const_in + noise_in
            # print('1',gen_in)
            gen_in = cond_in + noise_in
            print('2',gen_in)
            disc_in_real = sample_in[0]
            
            if self.mode == 'GAN':
                disc_in_fake = self.gen(gen_in) #TODO fix error here! Why is the shape wrong when you remove constants?
            elif self.mode == 'VAEGAN':
                encoder_in = cond_in + const_in
                (encoder_mean, encoder_log_var) = self.gen.encoder(encoder_in)
                decoder_in = [encoder_mean,encoder_log_var,noise_in,const_in]
                disc_in_fake = self.gen.decoder(decoder_in) 
            disc_in_avg = RandomWeightedAverage()([disc_in_real, disc_in_fake])
            # disc_out_real = self.disc(cond_in + const_in + [disc_in_real])
            # disc_out_fake = self.disc(cond_in + const_in + [disc_in_fake])
            # disc_out_avg = self.disc(cond_in + const_in + [disc_in_avg])
            disc_out_real = self.disc(cond_in +  [disc_in_real])
            disc_out_fake = self.disc(cond_in + [disc_in_fake])
            disc_out_avg = self.disc(cond_in + [disc_in_avg])


            print('disk out avg',disc_out_avg)
            print('disk in avg',disc_in_avg)
            print(GradientPenalty())
            disc_gp = GradientPenalty()([disc_out_avg, disc_in_avg]) #TODO: figure out why this doesn't work
            print('disc_gp',disc_gp)
            # print('const in ',const_in)
            # self.disc_trainer = Model(inputs=cond_in + const_in + noise_in + sample_in,
            #                           outputs=[disc_out_real, disc_out_fake, disc_gp], 
            #                           name='disc_trainer') # TODO: need to remove const and get everything in right shape
            self.disc_trainer = Model(inputs=cond_in + noise_in + sample_in, # remove const
                                      outputs=[disc_out_real, disc_out_fake, disc_gp], 
                                      name='disc_trainer')

        self.compile()

    def compile(self, opt_disc=None, opt_gen=None):
        #create optimizers
        if opt_disc is None:
            opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_disc = opt_disc
        if opt_gen is None:
            opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)
        self.opt_gen = opt_gen

        with Nontrainable(self.disc):
            if self.mode == 'GAN':
                self.gen_trainer.compile(loss=wasserstein_loss,
                                         optimizer=self.opt_gen)
            elif self.mode == 'VAEGAN':
                self.gen_trainer.compile(optimizer=self.opt_gen)
        with Nontrainable(self.gen):
            self.disc_trainer.compile(
                loss=[wasserstein_loss, wasserstein_loss, 'mse'], 
                loss_weights=[1.0, 1.0, self.gradient_penalty_weight],
                optimizer=self.opt_disc
            )
            self.disc_trainer.summary()

    def train(self, batch_gen, noise_gen, num_gen_batches=1, 
        training_ratio=1, show_progress=True):

        disc_target_real = None
        # for tmp_batch, _, _ in batch_gen.take(1).as_numpy_iterator(): # the 2nd argument is constant, whatever that is
        for tmp_batch, _ in batch_gen.take(1).as_numpy_iterator(): #have changed to this for now
            batch_size = tmp_batch.shape[0]
        del tmp_batch
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                num_gen_batches*batch_size)
        disc_target_real = np.ones(
            (batch_size, 1), dtype=np.float32)
        disc_target_fake = -disc_target_real
        gen_target = disc_target_real
        target_gp = np.zeros((batch_size, 1), dtype=np.float32)
        disc_target = [disc_target_real, disc_target_fake, target_gp]

        loss_log = []

        batch_gen_iter = iter(batch_gen)

        for k in range(num_gen_batches):
        
            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            for rep in range(training_ratio):
                # generate some real samples
                # (cond,const,sample) = batch_gen_iter.get_next()
                (cond,sample) = batch_gen_iter.get_next() # remove const as we don't have the constant features here
                print('training ratio', training_ratio)
                print('rep', rep)
                print('cond',cond.shape)
                print('sample',sample.shape)
                # print('disc_target',disc_target)
                print('noise gen',noise_gen)
                with Nontrainable(self.gen):   
                    dl = self.disc_trainer.train_on_batch(
                        # [cond,const,noise_gen(),sample], disc_target) #TODO: figure out what const is
                        [cond,noise_gen(),sample], disc_target)
                        # TODO: perhaps change input shape as we only have 1 feature currently

                if disc_loss is None:
                    disc_loss = np.array(dl)
                else:
                    disc_loss += np.array(dl)
                disc_loss_n += 1

                # del sample, cond, const
                del sample, cond

            disc_loss /= disc_loss_n

            with Nontrainable(self.disc):
                # (cond, const, sample) = batch_gen_iter.get_next()
                (cond, sample) = batch_gen_iter.get_next()
                if self.mode == 'GAN':
                    gen_loss = self.gen_trainer.train_on_batch(
                        # [cond,const,noise_gen()], gen_target)
                        [cond,noise_gen()], gen_target)
                elif self.mode == 'VAEGAN':
                    gen_loss = self.gen_trainer.train_step(
                        # [[cond,const,noise_gen()], gen_target])
                        [[cond,noise_gen()], gen_target])
                # del sample, cond, const
                del sample, cond
                
            if show_progress:
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                if self.mode == 'GAN':
                    for (i,gl) in enumerate([gen_loss]):
                        losses.append(("G{}".format(i), gl))
                elif self.mode == 'VAEGAN':
                    for (i,gl) in enumerate(gen_loss):
                        losses.append(("G{}".format(i), gl))
                progbar.add(batch_size, 
                    values=losses)

            loss_log.append(np.hstack((disc_loss,gen_loss)))

            gc.collect()

        return np.array(loss_log)
