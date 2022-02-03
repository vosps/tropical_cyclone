import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

sns.set_style("white")

def plot_predictions(real,pred,inputs,plot='save',mode='validation'):
        real[real<=0.1] = np.nan
        pred[pred<=0.1] = np.nan
        # inputs = regrid(inputs[99])
        inputs[inputs<=0.1] = np.nan
        n = 4
        m = 3
        if plot == 'save':
                fig, axes = plt.subplots(n, m, figsize=(5*m, 5*n), sharey=True)
        else:
                print('show')
                fig, axes = plt.subplots(n, m, figsize=(2*m, 2*n), sharey=True)
        range_ = (-5, 20)
        if mode == 'gcm':
                range_ = (-5,30)

        storms = [102,260,450,799]
        if mode == 'gcm':
                storms = [0,4,2,3,4]
        axes[0,0].set_title('Real')
        axes[0,1].set_title('Pred')
        axes[0,2].set_title('Input')
        for i in range(n):
                j = 0
                storm = storms[i]
                axes[i,j].imshow(real[storm], interpolation='nearest', norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+1].imshow(pred[storm], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+2].imshow(regrid(inputs[storm]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j].set(xticklabels=[])
                axes[i,j].set(yticklabels=[])
                axes[i,j+1].set(xticklabels=[])
                axes[i,j+1].set(yticklabels=[])
                axes[i,j+2].set(xticklabels=[])
                axes[i,j+2].set(yticklabels=[])

        if plot == 'save':
                plt.savefig('figs/pred_images_%s.png' % mode,bbox_inches='tight')
                plt.clf()
        else:
                plt.show()

def regrid(array):
        hr_array = np.zeros((100,100))
        for i in range(10):
                for j in range(10):
                        i1 = i*10
                        i2 = (i+1)*10
                        j1 = j*10
                        j2 = (j+1)*10
                        hr_array[i1:i2,j1:j2] = array[i,j]
        return hr_array