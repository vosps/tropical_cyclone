import os
from string import ascii_lowercase
from matplotlib import pyplot as plt
from matplotlib import colors, gridspec
import numpy as np
import pandas as pd
import data
from noise import NoiseGenerator

path = os.path.dirname(os.path.abspath(__file__))

def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.Normalize(*value_range), extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

def plot_img_log(img, value_range=(0.01, 5), extent=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.LogNorm(*value_range), extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

def plot_sequences(gen, 
                   mode, 
                   batch_gen,
                   noise_shape,
                   epoch,
                   num_samples=8, 
                   num_instances=4, 
                   out_fn=None):
    
    for cond, const,  seq_real in batch_gen.take(1).as_numpy_iterator():
        batch_size = cond.shape[0]

    seq_gen = []
    if mode == 'GAN':
        for i in range(num_instances):
            noise_in = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.predict([cond, const, noise_in]))
    elif mode == 'det':
        for i in range(num_instances):
            seq_gen.append(gen.predict([cond, const]))
    elif mode == 'VAEGAN':
        ## call encoder
        (mean, logvar) = gen.encoder([cond, const])
        ## run decoder n times
        for i in range(num_instances):
            noise_in = NoiseGenerator(noise_shape, batch_size=batch_size)
            seq_gen.append(gen.decoder.predict([mean, logvar, noise_in, const]))

    seq_real = data.denormalize(seq_real)
    cond = data.denormalize(cond)
    seq_gen = [data.denormalize(seq) for seq in seq_gen]

    num_rows = num_samples
    num_cols = 2+num_instances

    figsize = (num_cols*1.5, num_rows*1.5)
    plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(num_rows, num_cols, 
        wspace=0.05, hspace=0.05)

    value_range = (0,1)# batch_gen.decoder.value_range

    for s in range(num_samples):
        i = s
        plt.subplot(gs[i,0])
        plot_img(seq_real[s,:,:,0], value_range=value_range)
        plt.subplot(gs[i,1])
        plot_img(cond[s,:,:,0], value_range=value_range)
        for k in range(num_instances):
            j = 2+k
            plt.subplot(gs[i,j])
            plot_img(seq_gen[k][s,:,:,0], value_range=value_range) 

    plt.suptitle('Epoch ' + str(epoch))

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()


def plot_rank_metrics_by_samples(metrics_fn,ax=None,
    plot_metrics=["KS", "DKL", "OP", "mean"], value_range=(-0.1,0.2),
    linestyles=['solid', 'dashed', 'dashdot', ':',],
    opt_switch_point=350000, plot_switch_text=True):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(metrics_fn, delimiter=" ")

    x = df["N"]
    for (metric,linestyle) in zip(plot_metrics,linestyles):
        y = df[metric]
        label = metric
        if metric=="DKL":
            label = "$D_\\mathrm{KL}$"
        if metric=="OP":
            label = "OF"
        if metric=="mean":
            y = y-0.5
            label = "mean - $\\frac{1}{2}$"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0,x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    ax.axvline(opt_switch_point, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)
    if plot_switch_text:
        text_x = opt_switch_point*0.98
        text_y = value_range[1]-(value_range[1]-value_range[0])*0.02
        ax.text(text_x, text_y, "Adam\u2192SGD", horizontalalignment='right',
            verticalalignment='top', color=(0.5,0.5,0.5))
    plt.grid(axis='y')


def plot_rank_metrics_by_samples_multiple(metrics_files,
    value_ranges=[(-0.025,0.075),(-0.1,0.2)]):
    (fig,axes) = plt.subplots(len(metrics_files),1, sharex=True,
        squeeze=True)
    plt.subplots_adjust(hspace=0.1)

    for (i,(ax,fn,vr)) in enumerate(zip(axes,metrics_files,value_ranges)):
        plot_rank_metrics_by_samples(fn,ax,plot_switch_text=(i==0),value_range=vr)
        if i==len(metrics_files)-1:
            ax.legend(ncol=5)
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
        ax.set_ylabel("Rank metric")
        ax.grid(axis='y')


def plot_quality_metrics_by_samples(quality_metrics_fn,
    rank_metrics_fn, ax=None,
    plot_metrics=["RMSE", "MSSSIM", "LSD", "CRPS"], value_range=(0,0.7),
    linestyles=['-', '--', ':', '-.']):

    if ax is None:
        ax = plt.gca()

    df = pd.read_csv(quality_metrics_fn, delimiter=" ")
    df_r = pd.read_csv(rank_metrics_fn, delimiter=" ")
    df["CRPS"] = df_r["CRPS"]

    x = df["N"]
    for (metric,linestyle) in zip(plot_metrics,linestyles):
        y = df[metric]
        label = metric
        if metric=="MSSSIM":
            y = 1-y
            label = "$1 - $MS-SSIM"
        if metric=="LSD":
            label = "LSD [dB] / 50"
            y = y/50
        if metric=="CRPS":
            y = y*10
            label = "CRPS $\\times$ 10"
        ax.plot(x, y, label=label, linestyle=linestyle)

    ax.set_xlim((0,x.max()))
    ax.set_ylim(value_range)
    ax.axhline(0, linestyle='--', color=(0.75,0.75,0.75), zorder=-10)

def plot_quality_metrics_by_samples_multiple(
    quality_metrics_files, rank_metrics_files):

    (fig,axes) = plt.subplots(len(quality_metrics_files),1, sharex=True,
        squeeze=True)
    plt.subplots_adjust(hspace=0.1)
    value_ranges = [(0,0.4),(0,0.8)]

    for (i,(ax,fn_q,fn_r,vr)) in enumerate(zip(
        axes,quality_metrics_files,rank_metrics_files,value_ranges)):
        plot_quality_metrics_by_samples(fn_q,fn_r,ax,
            plot_switch_text=(i==0), value_range=vr)
        if i==0:
            ax.legend(mode='expand', ncol=4, loc='lower left')
        if i==1:
            ax.set_xlabel("Training sequences")
        ax.text(0.04, 0.97, "({})".format(ascii_lowercase[i]),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
        ax.set_ylabel("Quality metric")
        ax.grid(axis='y')

def plot_rank_histogram(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0,1,N_ranks)
    db = (bc[1]-bc[0])
    bins = bc-db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h,_) = np.histogram(ranks,bins=bins)
    h = h / h.sum()

    ax.plot(bc,h,**plot_params)


def plot_rank_cdf(ax, ranks, N_ranks=101, **plot_params):

    bc = np.linspace(0,1,N_ranks)
    db = (bc[1]-bc[0])
    bins = bc-db/2
    bins = np.hstack((bins, bins[-1]+db))
    (h,_) = np.histogram(ranks,bins=bins)
    h = h.cumsum()
    h = h / h[-1]

    ax.plot(bc,h,**plot_params)


def plot_rank_histogram_all(rank_files, labels, log_path, name, N_ranks=101):
    
    (fig,axes) = plt.subplots(2,1,sharex=True,figsize=(6,3))
    plt.subplots_adjust(hspace=0.15)

    linestyles = ["-","--"]
    colors = ["C0", "C1"]

    for (fn_valid, label, ls, c) in zip (rank_files, labels, linestyles, colors):
        with np.load(fn_valid) as f:
            ranks = f['arr_0']
        plot_rank_histogram(axes[0], ranks, N_ranks=N_ranks,label=label, linestyle=ls, linewidth=0.75, c=c, zorder=2)

    bc = np.linspace(0,1,N_ranks)
    axes[0].plot(bc, [1./N_ranks]*len(bc), linestyle=':', label="Uniform", c='dimgrey', zorder=0)
    axes[0].set_ylabel("Norm. occurrence")
    ylim = axes[0].get_ylim()
    axes[0].set_ylim((0,ylim[1]))
    axes[0].set_xlim((0,1))
    axes[0].text(0.01, 0.97, "(a)",
        horizontalalignment='left', verticalalignment='top',
        transform=axes[0].transAxes)

    for (fn_valid, label, ls, c) in zip (rank_files, labels, linestyles, colors):
        with np.load(fn_valid) as f:
            ranks = f['arr_0']
        plot_rank_cdf(axes[1], ranks, N_ranks=N_ranks, label=label, linestyle=ls, linewidth=0.75, c=c, zorder=2)

    axes[1].plot(bc,bc,linestyle=':', label="Uniform", c='dimgrey', zorder=0)
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("Normalized rank")
    axes[1].set_ylim(0,1.1)
    axes[1].set_xlim((0,1))
    axes[1].text(0.01, 0.97, "(b)",
        horizontalalignment='left', verticalalignment='top',
        transform=axes[1].transAxes)
    axes[1].legend(loc='lower right')
    
    plt.savefig("{}/rank-distribution-{}.pdf".format(log_path, name), bbox_inches='tight')
    plt.close()

def gridplot(models, model_labels=None,
             vmin = 0, vmax = 1):
    nx=models[0].shape[0]
    ny=len(models)
    fig=plt.figure(dpi=200,figsize=(nx,ny))
    gs1 = gridspec.GridSpec(ny, nx)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

    for i in range(nx):
        for j in range(ny):
            # print(i,j)
            ax = plt.subplot(gs1[i+j*nx])# plt.subplot(ny,nx,i+1+j*nx)
            ax.pcolormesh(models[j][i,:,:],vmin=vmin,vmax=vmax)
            # ax.axis('off')
            ax.set(xticks=[], yticks=[])
            if i == 0 and (model_labels is not None):
                ax.set_ylabel(model_labels[j])
            ax.axis('equal')
    fig.text(0.5, 0.9, 'Dates', ha='center')
    fig.text(0.04, 0.5, 'Models', va='center', rotation='vertical')
    return
