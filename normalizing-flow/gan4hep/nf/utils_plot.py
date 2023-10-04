"""
Plotting functions for Normalizing Flow
"""
import os
import time
import re
import vector

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

fontsize=16
minor_size=14
leg_size=12
    
    
def compare(predictions, truths, img_dir, epoch, x_range, labels, nphotons, titles=None, scale=None, save_path=None, show=False, save=True): 
    idxs = nphotons * 3
    nrow = nphotons
    ncol = 3
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4.5*nrow), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    if not titles:
        titles = [''] * idxs
    
    for idx in range(idxs):
        ax = axs[idx]
        if scale is not None:
            x_range_idx = [x_range[0] * scale[idx], x_range[1] * scale[idx]]
        else:
            x_range_idx = x_range
        bmin, bmax = x_range_idx 
        
        x = truths[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals, _, _ = ax.hist(x, bins=50, range=x_range_idx, label='Target', **config, weights=np.ones(x.shape[0])/x.shape[0])
        max_y = np.max(yvals) * 1.1
        
        x = predictions[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        ax.hist(x, bins=50, range=x_range_idx, label='CNF', **config, weights=np.ones(x.shape[0])/x.shape[0])
        ax.set_ylim(0, max_y)
        
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel(labels[idx], fontsize=fontsize)
        if nphotons == 1:
            ax.set_ylabel('Proportion of photons', fontsize=fontsize)
        else:
            ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.set_title(titles[idx], fontsize=fontsize)
        ax.legend(fontsize=leg_size)

    if not save_path:
        save_path = os.path.join(img_dir, 'image_epoch_{:04d}_min{}_max{}.png'.format(epoch, bmin, bmax))
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')

    
def compare_feat(predictions, truths, img_dir, epoch, labels, nphotons, logET=False, titles=None, save_path=None, show=False, save=True):
    idxs = nphotons * 3
    nrow = nphotons
    ncol = 3
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 4.5*nrow), constrained_layout=True)
    axs = axs.flatten()
    config = dict(histtype='step', lw=2)
    if titles is None:
        titles = [''] * idxs
    legend_loc = ['best', 'best', 'lower center']
    
    ph1_bins = []
    for idx in range(ncol):
        ax = axs[idx]
        
        if idx == 0: # pT
            x_range = [0, 500]    
        elif idx == 1: # eta
            x_range = [-2.5, 2.5]
            # x_range = [-np.pi, np.pi]
        elif idx == 2: # phi
            x_range = [-np.pi, np.pi]
            # x_range = [-2.5, 2.5]
        else:
            x_range = [int(min(truths[:, idx]))-1, int(max(truths[:, idx]))+1]
        bmin, bmax = x_range
        
        x = truths[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals, bins, _ = ax.hist(x, bins=50, range=x_range, label='Target', **config, weights=np.ones(truths.shape[0])/truths.shape[0])
        
        x = predictions[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        ax.hist(x, bins=bins, label='CNF', **config, weights=np.ones(predictions.shape[0])/predictions.shape[0])
        
        ph1_bins.append(bins)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel(labels[idx], fontsize=fontsize)
        if nphotons == 1:
            ax.set_ylabel('Proportion of photons', fontsize=fontsize)
        else:
            ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.set_title(titles[idx], fontsize=fontsize)
        ax.legend(loc=legend_loc[idx], fontsize=leg_size)
        
        if logET and idx == 0:
            ax.set_yscale('log')
        
    for idx in range(ncol, idxs):
        ax = axs[idx]
        bins = ph1_bins[idx%ncol]
        yvals, bins, _ = ax.hist(truths[:, idx], bins=bins, label='Target', **config, weights=np.ones(truths.shape[0])/truths.shape[0])
        ax.hist(predictions[:, idx], bins=bins, label='CNF', **config, weights=np.ones(predictions.shape[0])/predictions.shape[0])
        
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel(labels[idx], fontsize=fontsize)
        if nphotons == 1:
            ax.set_ylabel('Proportion of photons', fontsize=fontsize)
        else:
            ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.set_title(titles[idx], fontsize=fontsize)
        ax.legend(loc=legend_loc[idx % ncol], fontsize=leg_size)
        
        if logET and idx % ncol == 0:
            ax.set_yscale('log')

    if not save_path:
        save_path = os.path.join(img_dir, 'image_epoch_{:04d}_feat.png'.format(epoch))
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    

def plot_corr(x, img_dir, epoch, labels, save_path=None, show=False, save=True):
    import seaborn as sns
    corr = pd.DataFrame(x).corr()
    if len(labels) == 6:
        plt.figure(figsize=(12, 6))
    plt.rcParams['axes.unicode_minus'] = False
    g = sns.heatmap(corr, annot=True, vmax=1, vmin=-1, center=0, cmap='bwr', fmt='.4f', annot_kws={"fontsize": minor_size});
    ax = g.axes
    ax.tick_params(width=2, grid_alpha=0.5, labelsize=leg_size)
    ax.set_xticklabels(labels, rotation = 0)
    ax.set_yticklabels(labels, rotation = 0);
    
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=leg_size)
    
    if not save_path:
        save_path = os.path.join(img_dir, 'corr_epoch_{:04d}.png'.format(epoch))
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')
    

def compare_scatter(predictions, truths, img_dir, epoch, labels, nphotons, save_path=None, show=False, save=True):
    idxs = nphotons * 3
    nrow = nphotons
    ncol = 3
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow), constrained_layout=True)
    axs = axs.flatten()

    for idx in range(idxs):
        ax = axs[idx]
        ax.scatter(truths[:, idx], predictions[:, idx], s=5, alpha=0.5)
        ax.set_xlabel('Target')
        ax.set_ylabel('CNF')
        ax.set_title(labels[idx])
    
    if not save_path:
        save_path = os.path.join(img_dir, 'scatter_epoch_{:04d}.png'.format(epoch))
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')

    
def plot_logfile(filename, epoch, img_dir, lr_filename=None):
    df = pd.read_csv(filename)
    df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te']
    df = df[df['epoch'].str.contains(r'\*')]
    if len(df) == 0:
        return
    
    df['epoch'] = df['epoch'].apply(lambda x: x[2:])
    df = df.reset_index(drop = True).astype('float64')
    
    # plot best Wasserstein distance so far across epochs
    wass_dist_best = []
    best_so_far = np.inf
    for wd in df['wass_dist']:
        if wd < best_so_far:
            best_so_far = wd
        wass_dist_best.append(best_so_far)
    df['wass_dist'] = wass_dist_best
    
    time = []
    total = 0
    run_total = 0
    for t in df['time']:
        if t < run_total:
            total += run_total
        run_total = t
        time.append(t + total)
    df['time'] = time
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    
    fontsize = 16
    minor_size = 14
    y_labels = ['Time[s]', 'Best Wasserstein Distance', 'Training Loss']
    y_data   = ['time', 'wass_dist', 'loss_tr']
    x_label = 'Epoch'
    x_data = 'epoch'
    for ib, values in enumerate(zip(y_data, y_labels)):
        ax = axs[ib]
        df.plot(x=x_data, y=values[0], ax=ax)
        ax.set_ylabel(values[1], fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size, right=True, top=True)
        ax.get_legend().remove()
        
    if lr_filename:
        df = pd.read_csv(lr_filename)
        df = df.rename(columns = {'# Epoch':'epoch'})
        df = df[df['epoch'].str.contains(r'\*')]
        df['epoch'] = df['epoch'].apply(lambda x: x[2:])
        df = df.reset_index(drop = True).astype('float64')
        
        ax = axs[3]
        nbatches = int(df.columns[-1])+1
        nepochs = df.shape[0]
        x = np.arange(0, nepochs*nbatches)
        ax.plot(x/nbatches, df.iloc[:, 1:].values.flatten())
        
        ax.set_ylabel('Learning Rate', fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size, right=True, top=True)
    
    plt.savefig(os.path.join(img_dir, 'train_epoch_{:04d}.png'.format(epoch)))
    plt.close('all')

    
# ============ Plotting functions for yy variables =============

def compute_yy(x):
    ph1 = vector.obj(pt=x['ph_pt1'], phi=x['ph_phi1'], eta=x['ph_eta1'], mass=0)
    ph2 = vector.obj(pt=x['ph_pt2'], phi=x['ph_phi2'], eta=x['ph_eta2'], mass=0)
    ph = ph1 + ph2
    return ph.mass, ph.pt


def plot_yy_cond(predictions, testing_truth, test_other, test_in, labels, labels_other, scale, img_dir, epoch, nphotons, save_path=None, show=False, save=True):
    pred_scaled, truths_scaled, other_vars, pred_feat, truth_feat, test_in = undo_scaling_cond(predictions, testing_truth, test_other, test_in, labels, labels_other, scale)
    
    labels_feat = ['ph_pt1', 'ph_phi1', 'ph_eta1', 'ph_pt2', 'ph_phi2', 'ph_eta2']
    save_path = os.path.join(img_dir, 'corr_feat_epoch_{:04d}_pred.png'.format(epoch))
    plot_corr(pred_feat, img_dir, epoch, labels_feat, save_path=save_path, show=show, save=save)
    if epoch == 0:
        save_path = os.path.join(img_dir, 'corr_feat_epoch_{:04d}_truth.png'.format(epoch))
        plot_corr(truth_feat, img_dir, epoch, labels_feat, save_path=save_path, show=show, save=save)
    compare_feat(pred_feat.to_numpy(), truth_feat.to_numpy(), img_dir, epoch, labels_feat, nphotons, show=show, save=save)
    print()
    
    truth_feat.columns = labels_feat
    pred_feat.columns = labels_feat
    
    truth_yy = truth_feat.apply(lambda x: compute_yy(x), axis=1)
    truth_feat['myy'] = truth_yy.apply(lambda x: x[0])
    truth_feat['yy_pt'] = truth_yy.apply(lambda x: x[1])
    
    pred_yy = pred_feat.apply(lambda x: compute_yy(x), axis=1)
    pred_feat['myy'] = pred_yy.apply(lambda x: x[0])
    pred_feat['yy_pt'] = pred_yy.apply(lambda x: x[1])

    plot_myy(pred_feat, truth_feat, img_dir, epoch, show=show, save=save)
    plot_yy_pt(pred_feat, truth_feat, img_dir, epoch, show=show, save=save)


def plot_yy(predictions, testing_truth, test_other, labels, labels_other, scale, img_dir, epoch, save_path=None, show=False, save=True):
    pred_scaled, truths_scaled, other_vars = undo_scaling(predictions, testing_truth, test_other, labels, labels_other, scale)
    print()
    
    labels_feat = ['ph_pt1', 'ph_phi1', 'ph_eta1', 'ph_pt2', 'ph_phi2', 'ph_eta2']
    truths_scaled.columns = labels_feat
    pred_scaled.columns = labels_feat
    
    truth_yy = truths_scaled.apply(lambda x: compute_yy(x), axis=1)
    truths_scaled['myy'] = truth_yy.apply(lambda x: x[0])
    truths_scaled['yy_pt'] = truth_yy.apply(lambda x: x[1])
    
    pred_yy = pred_scaled.apply(lambda x: compute_yy(x), axis=1)
    pred_scaled['myy'] = pred_yy.apply(lambda x: x[0])
    pred_scaled['yy_pt'] = pred_yy.apply(lambda x: x[1])
    
    plot_yy_pt(pred_scaled, truths_scaled, img_dir, epoch, show=show, save=save)
    plot_myy(pred_scaled, truths_scaled, img_dir, epoch, show=show, save=save)
    
    
def undo_scaling_cond(predictions, testing_truth, test_other, test_in, labels, labels_other, scale):
    pred_scaled = pd.DataFrame(predictions, columns = labels)
    truths_scaled = pd.DataFrame(testing_truth, columns = labels)
    other_vars = pd.DataFrame(test_other, columns = labels_other)
    if scale:
        branch_scale, truth_in_scale, input_data_scale = scale
        pred_scaled = pred_scaled * truth_in_scale
        truths_scaled = truths_scaled * truth_in_scale
        test_in = test_in * input_data_scale.to_numpy()
        pred_feat = test_in[:, :6] + pred_scaled
        truth_feat = test_in[:, :6] + truths_scaled
    return pred_scaled, truths_scaled, other_vars, pred_feat, truth_feat, test_in
    
    
def undo_scaling(predictions, testing_truth, test_other, labels, labels_other, scale):
    pred_scaled = pd.DataFrame(predictions, columns = labels)
    truths_scaled = pd.DataFrame(testing_truth, columns = labels)
    other_vars = pd.DataFrame(test_other, columns = labels_other)
    if scale:
        branch_scale, truth_in_scale, input_data_scale = scale
        pred_scaled = pred_scaled * truth_in_scale
        truths_scaled = truths_scaled * truth_in_scale
        for b in branch_scale.keys():
            pred_scaled[b] = np.exp(pred_scaled[b] + branch_scale[b])
            truths_scaled[b] = np.exp(truths_scaled[b] + branch_scale[b])
    return pred_scaled, truths_scaled, other_vars


def plot_yy_pt(pred_scaled, truths_scaled, img_dir, epoch, show=False, save_path=None, save=True, plot_ratio=False, cond_scaled=None):
    if 'yy_pt' not in truths_scaled.columns:
        truth_yy = truths_scaled.apply(lambda x: compute_yy(x), axis=1)
        truths_scaled['myy'] = truth_yy.apply(lambda x: x[0])
        truths_scaled['yy_pt'] = truth_yy.apply(lambda x: x[1])
    if 'yy_pt' not in pred_scaled.columns:
        pred_yy = pred_scaled.apply(lambda x: compute_yy(x), axis=1)
        pred_scaled['myy'] = pred_yy.apply(lambda x: x[0])
        pred_scaled['yy_pt'] = pred_yy.apply(lambda x: x[1])
    if (cond_scaled is not None) and ('yy_pt' not in cond_scaled.columns):
        cond_yy = cond_scaled.apply(lambda x: compute_yy(x), axis=1)
        cond_scaled['myy'] = cond_yy.apply(lambda x: x[0])
        cond_scaled['yy_pt'] = cond_yy.apply(lambda x: x[1])
    
    lim = 1000
    bins = np.arange(0, lim+1, 25)
    plot_truth = truths_scaled['yy_pt'].copy()
    plot_pred = pred_scaled['yy_pt'].copy()
    plot_truth[plot_truth > lim] = lim
    plot_pred[plot_pred > lim] = lim
    if cond_scaled is not None:
        plot_cond = cond_scaled['yy_pt'].copy()
        plot_cond[plot_cond > lim] = lim
    
    if plot_ratio:
        nrow = 2
        ncol = 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow), constrained_layout=True)
        axs = axs.flatten()

        hist_truth_yypt,_,_ = axs[0].hist(plot_truth, bins = bins, label = 'Target', weights=np.ones(plot_truth.shape[0])/plot_truth.shape[0], histtype = 'step', linewidth = 2)
        hist_pred_yypt,_,_ = axs[0].hist(plot_pred, bins = bins, label = 'CNF', weights=np.ones(plot_pred.shape[0])/plot_pred.shape[0], ec='tab:orange', histtype = 'step', linewidth = 2)
        axs[0].tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        axs[0].set_xlabel('${p_T}^{\gamma\gamma}$ [GeV]', fontsize=fontsize)
        axs[0].set_ylabel('Proportion of events', fontsize=fontsize)
        axs[0].legend(fontsize=minor_size);
    
        n = len(hist_pred_yypt)
        hist_pred_yypt[hist_truth_yypt == 0] = 0
        hist_truth_yypt[hist_truth_yypt == 0] = 1
        axs[1].tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        axs[1].plot(hist_pred_yypt / hist_truth_yypt, label = 'CNF / Target', marker='o', ms=5);
        axs[1].set_xticks(ticks=np.arange(0,n+1,8), labels=bins[::8])
        axs[1].yaxis.grid(True, c='lightgray')
        axs[1].legend(fontsize=minor_size);
        
        if not save_path:
            save_path = os.path.join(img_dir, 'yy_pt_epoch_{:04d}_ratio.png'.format(epoch));
    else:
        plt.figure(figsize=(6, 4))
        plt.hist(plot_truth, bins = bins, label = 'Target', weights=np.ones(plot_truth.shape[0])/plot_truth.shape[0], histtype = 'step', linewidth = 2)
        plt.hist(plot_pred, bins = bins, label = 'CNF', weights=np.ones(plot_pred.shape[0])/plot_pred.shape[0], ec='tab:orange', histtype = 'step', linewidth = 2)
        if cond_scaled is not None:
            plt.hist(plot_cond, bins = bins, label = 'Truth', weights=np.ones(plot_cond.shape[0])/plot_cond.shape[0], ec='tab:green', histtype = 'step', linewidth = 2)
            
        ax = plt.gca()
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        plt.xlabel('${p_T}^{\gamma\gamma}$ [GeV]', fontsize=fontsize)
        plt.ylabel('Proportion of events', fontsize=fontsize)
        plt.legend(fontsize=minor_size);
        
        if not save_path:
            save_path = os.path.join(img_dir, 'yy_pt_epoch_{:04d}.png'.format(epoch))
        
    if save:
        plt.savefig(save_path, bbox_inches='tight');
    if show:
        plt.show()
    plt.close('all')
    
    # calculate mean, sd
    min_val = 0
    max_val = 400
    plot_truth = truths_scaled['yy_pt'].copy()
    plot_pred = pred_scaled['yy_pt'].copy()

    plot_truth = plot_truth[plot_truth < max_val].values
    plot_pred = plot_pred[plot_pred < max_val].values
    print(epoch, 'yy_pt')
    range_str = '['+str(min_val)+', '+str(max_val)+')'
    print('Range '+range_str+', # events in range, total # events, proportion of events in range')
    print('Target:', plot_truth.shape[0], truths_scaled.shape[0], plot_truth.shape[0]/truths_scaled.shape[0])
    print('CNF:', plot_pred.shape[0], pred_scaled.shape[0], plot_pred.shape[0]/pred_scaled.shape[0])
    
    print('Mean, sd for range '+range_str)
    print('Target:', plot_truth.mean(), plot_truth.std())
    print('CNF:', plot_pred.mean(), plot_pred.std())


def plot_myy(pred_scaled, truths_scaled, img_dir, epoch, show=False, save_path=None, save=True, plot_ratio=False):
    if 'myy' not in truths_scaled.columns:
        truth_yy = truths_scaled.apply(lambda x: compute_yy(x), axis=1)
        truths_scaled['myy'] = truth_yy.apply(lambda x: x[0])
        truths_scaled['yy_pt'] = truth_yy.apply(lambda x: x[1])
    if 'myy' not in pred_scaled.columns:
        pred_yy = pred_scaled.apply(lambda x: compute_yy(x), axis=1)
        pred_scaled['myy'] = pred_yy.apply(lambda x: x[0])
        pred_scaled['yy_pt'] = pred_yy.apply(lambda x: x[1])
        
    # calculate mean, sd
    min_val = 120
    max_val = 130
    plot_truth = truths_scaled['myy'].copy()
    plot_pred = pred_scaled['myy'].copy()

    plot_truth = plot_truth[(plot_truth > min_val) & (plot_truth < max_val)].values
    plot_pred = plot_pred[(plot_pred > min_val) & (plot_pred < max_val)].values
    print(epoch, 'myy')
    range_str = '('+str(min_val)+', '+str(max_val)+')'
    print('Range '+range_str+', # events in range, total # events, proportion of events in range')
    print('Target:', plot_truth.shape[0], truths_scaled.shape[0], plot_truth.shape[0]/truths_scaled.shape[0])
    print('CNF:', plot_pred.shape[0], pred_scaled.shape[0], plot_pred.shape[0]/pred_scaled.shape[0])
    
    print('Mean, sd for range '+range_str)
    print('Target:', plot_truth.mean(), plot_truth.std())
    print('CNF:', plot_pred.mean(), plot_pred.std())
    plot_truth_mean, plot_truth_sd = plot_truth.mean(), plot_truth.std()
    plot_pred_mean, plot_pred_sd = plot_pred.mean(), plot_pred.std()
    
    # make plot
    e = 0.01
    incr = 0.5
    min_val = 125-15
    max_val = 125+15
    bins = np.arange(min_val-incr, max_val+incr+e, incr)

    plot_truth = truths_scaled['myy'].copy()
    plot_pred = pred_scaled['myy'].copy()
    plot_truth[plot_truth > max_val] = max_val
    plot_truth[plot_truth < min_val] = min_val-e
    plot_pred[plot_pred > max_val] = max_val
    plot_pred[plot_pred < min_val] = min_val-e
    
    if plot_ratio:
        nrow = 2
        ncol = 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow), constrained_layout=True)
        axs = axs.flatten()

        hist_truth_myy,_,_ = axs[0].hist(plot_truth, bins = bins, label = 'Target', 
                                         weights=np.ones(plot_truth.shape[0])/plot_truth.shape[0], histtype = 'step', linewidth = 2)
        hist_pred_myy,_,_ = axs[0].hist(plot_pred, bins = bins, label = 'CNF', 
                                        weights=np.ones(plot_pred.shape[0])/plot_pred.shape[0], histtype = 'step', linewidth = 2)
        axs[0].tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        axs[0].set_xlabel('$m_{\gamma\gamma}$ [GeV]', fontsize=fontsize)
        axs[0].set_ylabel('Proportion of events', fontsize=fontsize)
        axs[0].legend(fontsize=minor_size);

        n = len(hist_pred_myy)
        hist_pred_myy[hist_truth_myy == 0] = 0
        hist_truth_myy[hist_truth_myy == 0] = 1
        axs[1].tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        axs[1].plot(hist_pred_myy / hist_truth_myy, label = 'CNF / Target', marker='o', ms=5);
        axs[1].set_xticks(ticks=np.arange(0,n,10), labels=bins[1::10].astype(int))
        axs[1].yaxis.grid(True, c='lightgray')
        axs[1].legend(fontsize=minor_size);
        
        ax = axs[0]
        if not save_path:
            save_path = os.path.join(img_dir, 'myy_epoch_{:04d}_ratio.png'.format(epoch));
    else:
        plt.figure(figsize = (6, 4))
        plt.hist(plot_truth, bins = bins, label = 'Target \nMean {:.3f} \nSD {:.3f}'.format(plot_truth_mean, plot_truth_sd),
                 weights=np.ones(plot_truth.shape[0])/plot_truth.shape[0], histtype = 'step', linewidth = 2)
        plt.hist(plot_pred, bins = bins, label = 'CNF \nMean {:.3f} \nSD {:.3f}'.format(plot_pred_mean, plot_pred_sd),
                 weights=np.ones(plot_pred.shape[0])/plot_pred.shape[0], histtype = 'step', linewidth = 2)
        
        ax = plt.gca()
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        plt.xlabel('$m_{\gamma\gamma}$ [GeV]', fontsize=fontsize)
        plt.ylabel('Proportion of events', fontsize=fontsize)
        plt.legend(fontsize=11.5);
        
        ax = plt.gca()
        if not save_path:
            save_path = os.path.join(img_dir, 'myy_epoch_{:04d}.png'.format(epoch))

    if save:
        plt.savefig(save_path, bbox_inches='tight');
    if show:
        plt.show()
    plt.close('all')
