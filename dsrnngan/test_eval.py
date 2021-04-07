import train
import noise
from eval import ensemble_ranks

batch_size = 16
val_size = 128
(wgan, _, batch_gen_valid, _,
 noise_shapes, _) = train.setup_gan(2015,2019,
                                    batch_size=batch_size,val_size=val_size)
noise_gen = noise.NoiseGenerator(noise_shapes(),
                                 batch_size=batch_size)
gen = wgan.gen

print("Loading real weights")
gen.load_weights('/ppdata/mat-cGAN/test-logs/gen_weights-ERA-0364800.h5')

print("Running ensemble ranks")
(ranks, crps_scores) = ensemble_ranks(gen, batch_gen_valid,
                                      noise_gen, 
                                      num_batches=val_size//batch_size)
print("Running ensemble ranks")
(ranks, crps_scores) = ensemble_ranks(gen, batch_gen_valid,
                                      noise_gen, 
                                      num_batches=val_size//batch_size)
