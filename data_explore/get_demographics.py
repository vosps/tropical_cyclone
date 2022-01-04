"""
This script plots some demographics of the TCs used in the training set.

	1. number of TCs/images per SS category
	2. number of TCs/images per basin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


df = pd.read_csv('/OLD/work/al18709/ibtracks/tc_files.csv',keep_default_na=False)


# tidy up columns with multiple dtypes


print(df.shape)
df['sshs'] = df['sshs'].astype(int)
cats = df['sshs'].value_counts(ascending=False)
basins = df['basin'].value_counts(ascending=False)

print(df['basin'].value_counts(ascending=False))

ax = cats.plot.bar(rot=0)
plt.xlabel('Saffir-Simpson Category')
plt.ylabel('Image Count')
plt.savefig('figs/categories.png')
plt.clf()

ax = basins.plot.bar(rot=0)
plt.xlabel('Basin')
plt.ylabel('Image Count')
plt.savefig('figs/basins.png')

