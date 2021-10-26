import gc
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import generic_utils
from meta import ensure_list, input_shapes
from meta import save_opt_weights, load_opt_weights


class Deterministic(object):

    def __init__(self, gen, lr, loss, optimizer):

        self.gen = gen
        self.lr = lr
        self.loss = loss
        self.optimizer = optimizer
        self.build_deterministic()

    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
        }
        return fn

    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.gen_trainer._make_train_function()
        load_opt_weights(self.gen_trainer, load_files["gen_opt_weights"])

    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])

    def build_deterministic(self):

        # find shapes for inputs
        cond_shapes = input_shapes(self.gen, "lo_res_inputs")
        const_shapes = input_shapes(self.gen, "hi_res_inputs")

        # Create generator training network
        cond_in = [Input(shape=s) for s in cond_shapes]
        const_in = [Input(shape=s) for s in const_shapes]
        gen_in = cond_in + const_in
        gen_out = self.gen(gen_in)
        gen_out = ensure_list(gen_out)
        self.gen_trainer = Model(inputs=gen_in, outputs=gen_out)

        self.gen_trainer.compile(loss=self.loss, optimizer=self.optimizer(lr=self.lr))

        self.gen_trainer.summary()

    def train(self, batch_gen_train, steps_per_epoch=1, show_progress=True):

        for tmp_batch, _, _ in batch_gen_train.take(1).as_numpy_iterator():
            batch_size = tmp_batch.shape[0]
        del tmp_batch

        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(steps_per_epoch*batch_size)

        loss_log = []

        batch_gen_iter = iter(batch_gen_train)

        for k in range(steps_per_epoch):
            cond, const, sample = batch_gen_iter.get_next()
            loss = self.gen_trainer.train_on_batch([cond, const], sample)
            del sample, cond, const

            if show_progress:
                losses = []
                for (i, l) in enumerate([loss]):
                    losses.append(("Loss {}".format(i), l))
                progbar.add(batch_size, values=losses)

            loss_log.append(loss)

            gc.collect()

        return np.array(loss_log)
