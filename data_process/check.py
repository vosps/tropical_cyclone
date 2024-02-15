import numpy as np
import pandas as pd

train = np.load("/user/work/al18709/tc_data_flipped/train_combined_X.npy")

print(train.shape)

meta = pd.read_csv('/user/work/al18709/tc_data_flipped/train_meta.csv')
print(meta.shape)

meta_with_dates = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_valid_meta_with_dates.csv')
print(meta_with_dates)