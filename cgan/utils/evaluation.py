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

def plot_anomaly(inputs,cmap,plot='save',vmin=-1,vmax=1,mode='validation'):  
        fig, ax = plt.subplots(figsize=(5, 5))  
        ax.set_title('Anomaly')
        im = ax.imshow(inputs, interpolation='nearest', vmin=vmin,vmax=vmax,extent=None,cmap=cmap)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
        # plt.colorbar(im)

        if plot == 'save':
                plt.savefig('figs/input_images_%s.png' % mode,bbox_inches='tight')
                plt.clf()
        else:
                plt.show()

def plot_histogram(real,pred,binwidth,alpha):
        """
        This function plots a histogram of the set in question
        """
        # ax = sns.histplot(data=penguins, x="flipper_length_mm", hue="species", element="step")
        fig, ax = plt.subplots()
        sns.histplot(ax=ax,data=real, stat="density", fill=True,color='#b5a1e2',element='step',alpha=alpha)
        sns.histplot(ax=ax,data=pred, stat="density", fill=True,color='#dc98a8',element='step',alpha=alpha)
        ax.set_xlabel('Accumulated rainfall (m)')
        plt.legend(labels=['real','pred'])
        plt.show()
        # plt.savefig('figs/histogram_accumulated_%s.png' % mode)

def calc_peak(array):
        nstorms,_,_ = array.shape
        peaks = np.zeros((nstorms))
        for i in range(nstorms):
                peaks[i] = np.nanmax(array[i])
        return peaks

def calc_mean(array):
        nstorms,_,_ = array.shape
        means = np.zeros((nstorms))
        for i in range(nstorms):
                means[i] = np.nanmean(array[i])
        return means