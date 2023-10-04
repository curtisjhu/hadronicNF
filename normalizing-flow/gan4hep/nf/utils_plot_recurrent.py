"""
Plotting functions for Normalizing Flow
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fontsize=16
minor_size=14
leg_size=12

file_type = 'png'
config = dict(histtype='step', lw=2)


def compare(predictions, truths, labels, nparticles, img_dir, epoch,
            x_range=None, log_scale=None, plot_ratio=False, titles=None, dataset='', save_path=None):    
    idxs = truths.shape[1]    
    if idxs % 3 == 1 or idxs % 4 == 0:
        nrow = int(np.ceil(idxs / 4))
    else:
        nrow = int(np.ceil(idxs / 3))
    ncol = int(np.ceil(idxs / nrow))
    
    if plot_ratio:
        fig, axs = plt.subplots(nrow*2, ncol, figsize=(6*ncol, 6*nrow), constrained_layout=True, 
                                gridspec_kw={'height_ratios': [2]*nrow + [1]*nrow})
    else:
        fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), constrained_layout=True)
    if nrow == 1 and ncol == 1 and not plot_ratio:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    if not x_range:
        x_range = [[-1, 1] for i in range(idxs)]
        scale_type = 'norm'
    else:
        scale_type = 'feat'
        
    if not log_scale:
        log_scale = [False] * idxs
    elif log_scale == True:
        log_scale = [True] * idxs
        
    if not titles:
        titles = [''] * idxs
    
    for idx in range(idxs):
        ax = axs[idx]
        num_bins = 40
        try:
            x_range_idx = x_range[idx]
        except:
            x_range_idx = [int(min(truths[:, idx]))-1, int(max(truths[:, idx]))+1]
            print('Inferring x-range for feature', idx, np.round(x_range, 2))
        bmin, bmax = x_range_idx
        
        x = truths[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals_sim, bins, _ = ax.hist(x, bins=num_bins, range=x_range_idx, label='Target', **config, weights=np.ones(x.shape[0])/x.shape[0])
        
        x = predictions[:, idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals_gen, _, _ = ax.hist(x, bins=bins, label='CNF', **config, weights=np.ones(x.shape[0])/x.shape[0])
        
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel(labels[idx], fontsize=fontsize)
        if nparticles == 1:
            ax.set_ylabel('Proportion of particles', fontsize=fontsize)
        else:
            ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.set_title(titles[idx], fontsize=fontsize)
        ax.legend(fontsize=leg_size)
        
        if log_scale[idx]:
            ax.set_yscale('log')
            bottom, top = ax.get_ylim()
            if bottom > 1e-4:
                ax.set_ylim(1e-4, top * 1.5)
            elif bottom > 1e-5:
                ax.set_ylim(1e-5, top * 1.5)
        else:
            max_y = np.max(yvals_sim) * 1.1
            ax.set_ylim(0, max_y)

        if plot_ratio:
            n = len(yvals_gen)
            yvals_gen[yvals_sim == 0] = 0
            yvals_sim[yvals_sim == 0] = 1
            ax = axs[nrow*ncol + idx]
            ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
            ax.plot((bins[:-1] + bins[1:]) / 2, yvals_gen / yvals_sim, label = 'CNF / Target', marker='o', ms=5, lw=2);
            ax.yaxis.grid(True, c='lightgray')
            ax.legend(fontsize=minor_size);
    
    if not save_path:
        save_path = 'image_epoch_{:04d}_{}'.format(epoch, scale_type)
        if dataset:
            save_path += '_' + dataset
        if any(log_scale):
            save_path += '_log'
        if plot_ratio:
            save_path += '_ratio'
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_particle_feat(predictions, pred_cond, pred_mask, truths, truths_cond, truths_mask, p, scale_truth,
                       labels, labels_in, idxs, cond_idxs, img_dir, epoch, x_range, dataset='', save_path=None, title=None):
    pred_mask = pred_mask[:, p]
    truths_mask = truths_mask[:, p]
    
    num_events = np.sum(pred_mask)
    num_events_truth = np.sum(truths_mask)
    print('Particle {}, generated {} events, simulated {} events'.format(p, num_events, num_events_truth))
    if num_events < 1000: # stop if too few events with p particles
        return
    
    x = predictions[:, p, :]
    predictions = x[pred_mask]
    pred_cond = pred_cond[:, p, :][pred_mask]
    x = truths[:, p, :]
    truths = x[truths_mask]
    truths_cond = truths_cond[:, p, :][truths_mask]
    
    nrow = 1
    ncol = len(cond_idxs) + len(idxs)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow), constrained_layout=True)
    axs = axs.flatten()
    
    # plot conditions
    weights_truth = np.ones(truths_cond.shape[0])/truths_cond.shape[0]
    weights_pred = np.ones(pred_cond.shape[0])/pred_cond.shape[0]
    
    for idx in range(len(cond_idxs)):
        ax = axs[idx]
        plot_idx = cond_idxs[idx]
        yvals_sim, bins, _ = ax.hist(truths_cond[:, plot_idx], bins=50, label='Target', weights=weights_truth, **config)
        max_y = np.max(yvals_sim) * 1.1
        yvals_gen, _, _ = ax.hist(pred_cond[:, plot_idx], bins=bins, label='CNF', weights=weights_pred, **config)
        ax.set_xlabel(labels_in[plot_idx], fontsize=fontsize)
        ax.set_ylim(0, max_y)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        if idx == 0:
            ax.set_ylabel('Proportion of particles', fontsize=fontsize)
    
    # plot output
    weights_truth = np.ones(truths.shape[0])/truths.shape[0]
    weights_pred = np.ones(predictions.shape[0])/predictions.shape[0]
    
    for idx in range(len(idxs)):
        ax = axs[idx + len(cond_idxs)]
        plot_idx = idxs[idx]
        scale_idx = scale_truth[plot_idx]
        bmin = x_range[0] * scale_idx
        bmax = x_range[1] * scale_idx
        
        num_bins = 50
        bins_range = [bmin, bmax]
        if idx == 0: # pt
            bins_range = None
        elif idx == 3: # particle label
            num_bins = 40 
            
        x = truths[:, plot_idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals_sim, bins, _ = ax.hist(x, bins=num_bins, range=bins_range, label='Target', weights=weights_truth, **config)
        max_y = np.max(yvals_sim) * 1.1
        
        x = predictions[:, plot_idx].copy()
        x[x < bmin] = bmin
        x[x > bmax] = bmax
        yvals_gen, _, _ = ax.hist(x, bins=bins, label='CNF', weights=weights_pred, **config)
        ax.set_xlabel(labels[plot_idx], fontsize=fontsize)
        ax.set_ylim(0, max_y)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    plt.legend(fontsize=leg_size)
    if title:
        fig.suptitle(title, fontsize=fontsize)
    if not save_path:
        save_path = 'image_epoch_{:04d}_feat_p{}'.format(epoch, p)
        if dataset:
            save_path += '_' + dataset
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight')
    
    
def plot_nparticles(num_particles, num_particles_test, img_dir, epoch, dataset='', save_path=None):
    weights_truth = np.ones(num_particles_test.shape[0])/num_particles_test.shape[0]
    weights_pred = np.ones(num_particles.shape[0])/num_particles.shape[0]

    plt.hist(num_particles_test, bins=np.arange(max(num_particles_test)+2), label='Target', weights=weights_truth, **config)
    plt.hist(num_particles, bins=np.arange(max(num_particles)+2), label='CNF', weights=weights_pred, **config)
    plt.xlabel('Number of particles', fontsize=fontsize)
    plt.ylabel('Proportion of events', fontsize=fontsize)
    plt.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    plt.legend(fontsize=leg_size)

    if not save_path:
        save_path = 'image_epoch_{:04d}_nparticles'.format(epoch)
        if dataset:
            save_path += '_' + dataset
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight');
    
    
def plot_particle_index(pred_mask, test_mask, img_dir, epoch, dataset='', save_path=None):
    num_particles_cond = pred_mask.sum(axis=0)
    num_particles_test_cond = test_mask.sum(axis=0)

    plt.plot(np.arange(test_mask.shape[1]), num_particles_test_cond / test_mask.shape[0], label='Target', marker='o')
    plt.plot(np.arange(pred_mask.shape[1]), num_particles_cond / pred_mask.shape[0], label='CNF', marker='o')
    plt.xlabel('Particle index', fontsize=fontsize)
    plt.ylabel('Proportion of events', fontsize=fontsize)
    plt.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    plt.legend(fontsize=leg_size)

    ax = plt.gca()
    ax.set_xticks(ticks=np.arange(0, max(test_mask.shape[1], pred_mask.shape[1]), 2))
    ax.yaxis.grid(True, c='lightgray')
    ax.xaxis.grid(True, c='lightgray')
    
    if not save_path:
        save_path = 'image_epoch_{:04d}_particle_index'.format(epoch)
        if dataset:
            save_path += '_' + dataset
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight');
    
    
def plot_nparticles_type(pred_label, pred_mask, test_label, test_mask, plabel_type, 
                         img_dir, epoch, types=None, dataset='', save_path=None):
    if types is None:
        types = np.unique(test_label)
    num_type = len(types)
    
    if num_type % 3 == 1 or num_type % 4 == 0:
        nrow = int(np.ceil(num_type / 4))
    else:
        nrow = int(np.ceil(num_type / 3))
    ncol = int(np.ceil(num_type / nrow))
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), constrained_layout=True)
    axs = axs.flatten()
    
    weights_truth = np.ones(test_label.shape[0])/test_label.shape[0]
    weights_pred = np.ones(pred_label.shape[0])/pred_label.shape[0]

    for plabel in types:
        ax = axs[plabel]
        num_label_test = np.sum((test_label == plabel) * test_mask, axis=1)
        num_label = np.sum((pred_label == plabel) * pred_mask, axis=1)

        ax.hist(num_label_test, bins=np.arange(max(num_label_test)+2), label='Target', weights=weights_truth, **config)
        ax.hist(num_label, bins=np.arange(max(num_label)+2), label='CNF', weights=weights_truth, **config)
        ax.set_xlabel('Number of ' + plabel_type[plabel], fontsize=fontsize)
        ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.legend(fontsize=leg_size)
    
    if not save_path:
        save_path = 'image_epoch_{:04d}_nparticles_type'.format(epoch)
        if dataset:
            save_path += '_' + dataset
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight');
    
        
def plot_multiparticle(x_plot, idxs_plot, img_dir, epoch, log_scale=None, plot_ratio=False, dataset='', save_path=None):
    if len(idxs_plot) % 3 == 1 or len(idxs_plot) % 4 == 0:
        nrow = int(np.ceil(len(idxs_plot) / 4))
    else:
        nrow = int(np.ceil(len(idxs_plot) / 3))
    ncol = int(np.ceil(len(idxs_plot) / nrow))
    
    if plot_ratio:
        fig, axs = plt.subplots(nrow*2, ncol, figsize=(6*ncol, 6*nrow), constrained_layout=True, 
                                gridspec_kw={'height_ratios': [2, 1]*nrow})
    else:
        fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 4*nrow), constrained_layout=True)
    if nrow == 1 and ncol == 1 and not plot_ratio:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    if not log_scale:
        log_scale = [False] * len(idxs_plot)
    elif log_scale == True:
        log_scale = [True] * len(idxs_plot)

    for idx in range(nrow * ncol):
        if plot_ratio:
            idx_ax = idx//ncol * ncol * 2 + idx % ncol
        else:
            idx_ax = idx
        if idx >= len(idxs_plot):
            axs[idx_ax].set_axis_off()
            if plot_ratio:
                axs[idx_ax + ncol].set_axis_off()
            continue
        
        ax = axs[idx_ax]
        x_truth, x, min_val, max_val, x_label = x_plot[idxs_plot[idx]]
        x_truth = np.array(x_truth)
        x = np.array(x)
        if min_val is not None:
            x_truth[x_truth < min_val] = min_val
            x[x < min_val] = min_val
        if max_val is not None:
            x_truth[x_truth > max_val] = max_val
            x[x > max_val] = max_val

        yvals_sim, bins, _ = ax.hist(x_truth, bins=50, label='Target', **config, weights=np.ones(x_truth.shape[0])/x_truth.shape[0])
        yvals_gen, _, _ = ax.hist(x, bins=bins, label='CNF', **config, weights=np.ones(x.shape[0])/x.shape[0])
        
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel(x_label, fontsize=fontsize);
        ax.set_ylabel('Proportion of events', fontsize=fontsize)
        ax.legend(fontsize=leg_size)
        
        if log_scale[idx]:
            ax.set_yscale('log')
            bottom, top = ax.get_ylim()
            if bottom > 1e-4:
                ax.set_ylim(1e-4, top * 1.5)
            elif bottom > 1e-5:
                ax.set_ylim(1e-5, top * 1.5)

        if plot_ratio:
            n = len(yvals_gen)
            yvals_gen[yvals_sim == 0] = 0
            yvals_sim[yvals_sim == 0] = 1
            
            ax = axs[idx_ax + ncol]
            ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
            ax.plot((bins[:-1] + bins[1:]) / 2, yvals_gen / yvals_sim, label = 'CNF / Target', marker='o', ms=5, lw=2);
            ax.yaxis.grid(True, c='lightgray')
            ax.legend(fontsize=minor_size);

    if not save_path:
        save_path = 'image_epoch_{:04d}_multiparticle'.format(epoch)
        if dataset:
            save_path += '_' + dataset
        if any(log_scale):
            save_path += '_log'
        if plot_ratio:
            save_path += '_ratio'
        save_path = os.path.join(img_dir, save_path + '.' + file_type)
    plt.savefig(save_path, bbox_inches='tight');

    
def plot_logfile(filename, epoch, img_dir, lr_filename=None, use_val=False, logy=False, cutoff=0):
    df = pd.read_csv(filename)
    if use_val:
        df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te', 'loss_va']
    else:
        df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te']
    df = df[df['epoch'].str.contains(r'\*')]
    if len(df) == 0:
        return
    df = df[cutoff:]
    
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
    y_labels = ['Time[s]', 'Best Wasserstein Distance', 'Training Loss vs Validation Loss']
    y_data   = ['time', 'wass_dist', 'loss_tr']
    x_label = 'Epoch'
    x_data = 'epoch'
    for ib, values in enumerate(zip(y_data, y_labels)):
        ax = axs[ib]
        df.plot(x=x_data, y=values[0], ax=ax, logy=logy)
        ax.set_ylabel(values[1], fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size, right=True, top=True)
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    if use_val:
        ax = axs[2]
        df.plot(x=x_data, y='loss_va', ax=ax, logy=logy)

        
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