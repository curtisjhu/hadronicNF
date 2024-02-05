import vector
import uproot
import math
import yaml
import pandas as pd
import numpy as np
import re
import os
import sys
import itertools
import configparser
import hickle as hkl


def load_yaml(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def read_data_root_hadronic(file_name, tree_name,
                            out_branch_names,
                            in_branch_names,
                            data_branch_names=None,
                            max_evts=None, test_frac=0.1, val_frac=0.1):
    
    if type(file_name) != list:
        file_name = [file_name]
    if not in_branch_names:
        in_branch_names = []
    if not data_branch_names:
        data_branch_names = []

    dfs = []
    branch_names = out_branch_names + in_branch_names + data_branch_names
    for f in file_name:
        in_file = uproot.open(f)
        tree = in_file[tree_name]
        array_root = tree.arrays(branch_names, library="np")
        df_root = pd.DataFrame(array_root)
        dfs += [df_root]
    df = pd.concat(dfs)

    truth = df[out_branch_names].to_numpy()
    nparts = np.array([np.shape(event[0])[0] for event in truth])

    m = nparts.max()

    arr_pad = []

    for i, event in enumerate(truth):
        to_pad = np.stack(event).T
        to_pad = to_pad[to_pad[:,1].argsort()[::-1]] #Sort by energy
        pad = np.zeros((m-nparts[i], 5))
        to_pad = np.concatenate((to_pad, pad))
        arr_pad.append(to_pad)
    arr_pad = np.array(arr_pad)

    mask = np.zeros(arr_pad.shape[:2])
    for i in range(mask.shape[0]):
        mask[i, :nparts[i]] = 1
    mask = mask.astype(bool)    

    print(np.shape(arr_pad))

    t_kine = arr_pad[:,:,1:]
    index = 1
    remaining = np.array([event for event in df["incoming"]])
    num_events = np.shape(remaining)[0]
    indices = index*np.ones(num_events)
    conds = [np.concatenate((remaining, indices[:, None]), axis=1)]
    #print(conds)
    for i in range(m-1):
        index += 1
        remaining = remaining - t_kine[:,i,:]
        #print(remaining)
        indices = index*np.ones(num_events)
        cond = np.concatenate((remaining, indices[:, None]), axis=1)
        conds.append(cond)

    conds = np.stack((conds), axis=1)
    print(np.shape(conds))

    total_parts = np.shape(conds)[0]*np.shape(conds)[1]

    t_pad_flat = np.reshape(arr_pad, (total_parts, 5))
    conds_flat = np.reshape(conds, (total_parts, 5))
    mask = np.reshape(mask, (total_parts,))

    truth_vectors = t_pad_flat[mask]
    cond_vectors = conds_flat[mask]

    cond_e = cond_vectors[:,0]
    truth_e = truth_vectors[:,1]
    scale_max = np.max(cond_e)
    scale_min = np.min(remaining[:,0])
    shift = scale_min*np.ones(np.shape(cond_e)[0])
    cond_e = cond_e - shift
    print(np.max(cond_e))
    print(scale_max)
    print(scale_min)
    scale_max = scale_max - scale_min

    ecv_i = cond_vectors[:,4]
    etv_id = truth_vectors[:,0]
    id_list = np.unique(etv_id)
    idx_list = range(np.shape(id_list)[0])
    partdict = dict(zip(id_list, idx_list))
    etv_id = np.array([partdict[id] for id in etv_id])
    etv_id = pscale(etv_id, np.shape(id_list)[0], 0)

    truth_vectors_no_id = truth_vectors[:,2:]
    cond_vectors_no_i = cond_vectors[:,1:4]
    truth_e = pscale(truth_e, scale_max, 0)
    truth_vectors_momentum = pscale(truth_vectors_no_id, scale_max, -scale_max)
    cond_e = pscale(cond_e, scale_max, 0)
    cond_vectors_momentum = pscale(cond_vectors_no_i, scale_max, -scale_max)

    cond_vectors = np.concatenate((cond_e[:, None], cond_vectors_momentum, ecv_i[:, None]), axis=1)
    truth_vectors = np.concatenate((etv_id[:, None], truth_e[:, None], truth_vectors_momentum), axis=1)


    scale = np.array([scale_max, scale_min])
    print("cmax: ", np.max(cond_vectors))
    print("cmin: ", np.min(cond_vectors))
    print("tmax: ", np.max(truth_vectors))
    print("tmin: ", np.min(truth_vectors))

    
    num_events = len(truth_vectors)
    if max_evts:
        if num_events > max_evts: num_events = max_evts
    if val_frac:
        test_num = math.floor(num_events*test_frac)
        val_num = math.floor(num_events*val_frac)
        test_truth = truth_vectors[:test_num]
        val_truth = truth_vectors[test_num:test_num + val_num]
        train_truth = truth_vectors[test_num + val_num:num_events]
        test_in = cond_vectors[:test_num]
        val_in = cond_vectors[test_num:test_num + val_num]
        train_in = cond_vectors[test_num + val_num:num_events]
        data = (train_in, train_truth, test_in, test_truth, val_in, val_truth)
    else:
        test_num = math.floor(num_events*test_frac)
        test_truth = truth_vectors[:test_num]
        train_truth = truth_vectors[test_num:num_events]
        test_in = cond_vectors[:test_num]
        train_in = cond_vectors[test_num:num_events]
        data = (train_in, train_truth, test_in, test_truth)
    scale = scale
    label = (in_branch_names, out_branch_names, data_branch_names)

    to_return = (data, scale, label, partdict)
    return to_return


# Recurrent NF
def read_data_root_recurrent(file_name, tree_name,
                             out_branch_names, # variables to generate
                             in_branch_names=None, # optional: conditional variables to input
                             data_branch_names=None, # optional: additional variables
                             model_type=-1, # specify conditional inputs
                             max_evts=None, max_train_evts=None, max_val_evts=None, max_test_evts=None, test_frac=0.1, sample='ttH'):
    if type(file_name) != list:
        file_name = [file_name]
    if not in_branch_names:
        in_branch_names = []
    if not data_branch_names:
        data_branch_names = []
    
    dfs = []
    branch_names = out_branch_names + in_branch_names + data_branch_names
    for f in file_name:
        in_file = uproot.open(f)
        tree = in_file[tree_name]
        array_root = tree.arrays(branch_names, library="np")
        df_root = pd.DataFrame(array_root)
        dfs += [df_root]
    df = pd.concat(dfs)

    if sample == 'delphes':
        cols_rename = {'genjet_pt':'pt', 'genjet_phi':'phi', 'genjet_eta':'eta',
                       'object_pt':'pt', 'object_phi':'phi', 'object_eta':'eta', 'object_tag':'label'}
        df = df.rename(columns = cols_rename)
        out_branch_names = [cols_rename[b] if b in cols_rename.keys() else b for b in out_branch_names]

        df['type'] = df['label'].apply(lambda x: convert_label_type(x))
        df['njet'] = df['type'].apply(lambda x: sum(x != 4))
        df = df[df['njet'] > 0]
        fold = (df['eta'].apply(lambda x: x[0]).astype('float64') * 10**6 % 10).astype(int).values
        # fold = (df['GenMET_phi'] * 1000 % 10).astype(int).values
    
    truth_in = df[in_branch_names].to_numpy()
    truth = df[out_branch_names].to_numpy()
    print(truth)
    truth_pad, truth_mask = pad_array(df, truth, ['njet', 'type'])
    
    truth_sum_ET = truth_pad[:,:,0].sum(axis=-1)
    truth_in = np.concatenate([np.expand_dims(truth_sum_ET, axis=-1), truth_in], axis=-1)
    in_branch_names = ['sum_ET'] + in_branch_names
    
    # drop event if any particle has pt beyond thresh_p
    thresh_p = 1000
    pt = truth_pad[:,:,0]
    drop = (np.sum(abs(pt) >= thresh_p, axis=1) != 0)
    truth_in = truth_in[~drop]
    truth_pad = truth_pad[~drop]
    truth_mask = truth_mask[~drop]

    truth_mask_int = truth_mask.astype(int)
    pt = truth_pad[:, :, 0] * truth_mask_int
    phi = truth_pad[:, :, 1] * truth_mask_int
    eta = truth_pad[:, :, 2] * truth_mask_int
    
    # particle
    num_events, num_particles = pt.shape
    dfs = []
    for i in range(num_particles):
        df = pd.DataFrame()
        df['pt'] = pt[:, i]
        df['phi'] = phi[:, i]
        df['eta'] = eta[:, i]
        df['m'] = 0
        dfs += [df]
    
    # calculate sum ET condition
    df_truth_in = pd.DataFrame(truth_in)
    for p in range(num_particles):
        df_truth_in[p+1] = df_truth_in[p] - pt[:, p]
    df_truth_in = df_truth_in.iloc[:, :-1] # last column is all zeros
    df_truth_in = df_truth_in * truth_mask_int
    truth_in = np.array(df_truth_in)
    truth_in = np.expand_dims(truth_in, -1)

    if model_type in [0, 1]:
        cols1 = ['pt1', 'phi1', 'eta1', 'm1']
        cols2 = ['pt2', 'phi2', 'eta2', 'm2']
        
        print('\nComputing multijet system: ')
        print('\tAdded particle:', 0)
        df1 = dfs[0].copy()
        df1.columns = cols1
        df0 = df1 * 0
        
        # existing multijet system
        dfs_multijet = [df1]
        for i in range(1, num_particles):
            df2 = dfs[i].copy()
            df2.columns = cols2
            df = pd.concat([df1, df2], axis=1)
            out = compute_multijet(df)
            out = np.array(out).T
            
            df1 = pd.DataFrame(out)
            df1.columns = cols1
            dfs_multijet += [df1]
            print('\tAdded particle:', i)
        print()
            
    if model_type in [2]:
        cols1 = ['pt1', 'phi1', 'eta1', 'm1']
        cols2 = ['pt2', 'phi2', 'eta2', 'm2']
    
        print('\nComputing multijet system: ')
        print('\tAdded particle:', 0)
        df1 = dfs[-1].copy()
        df1.columns = cols1
        df0 = df1 * 0

        # remaining multijet system
        dfs_multijet_reverse = [df1]
        for i in range(2, num_particles+1):
            df2 = dfs[-i].copy()
            df2.columns = cols2
            df = pd.concat([df1, df2], axis=1)
            out = compute_multijet(df)
            out = np.array(out).T
            
            df1 = pd.DataFrame(out)
            df1.columns = cols1
            dfs_multijet_reverse = [df1] + dfs_multijet_reverse
            print('\tAdded particle:', -i, -i+num_particles)
        print()

    # model_type: conditions
    # 0: sum ET (remaining) + multijet 4vec (existing) + previous particle pt/phi/eta
    # 1: sum ET (remaining) + multijet phi/eta (existing)
    # 2: sum ET (remaining) + mulijet m (remaining)
    if model_type == 0: 
        # dfs[0]: df_truth_in[[0]], df0, df0
        # dfs[1]: df_truth_in[[1]], dfs_multijet[0], dfs[0] where dfs_multijet[0] == dfs[0]
        # dfs[2]: df_truth_in[[2]], dfs_multijet[1], dfs[1]
        # ...

        i = 0
        df_sumET = df_truth_in[[i]]
        df = pd.concat([df_sumET, df0, df0], axis=1)
        truth_in_all = [np.array(df)]
        for i in range(1, num_particles):
            df_sumET = df_truth_in[[i]] # ['sum_ET']
            df_sum = dfs_multijet[i-1] # ['multijet_pt', 'multijet_phi', 'multijet_eta', 'multijet_m']
            df_prev = dfs[i-1] # ['pt', 'phi', 'eta', 'm']
            df = pd.concat([df_sumET, df_sum, df_prev], axis=1)
            truth_in_all += [np.array(df)]
        truth_in_all = np.array(truth_in_all)
        truth_in_all = np.transpose(truth_in_all, axes=[1, 0, 2])
        truth_in = truth_in_all[:,:,:-1] # exclude previous m, which is always 0
        in_branch_names = ['sum_ET', 'multiparticle_pt', 'multiparticle_phi', 'multiparticle_eta', 'multiparticle_m', 
                           'previous_pt', 'previous_phi', 'previous_eta']
    
    elif model_type == 1:
        i = 0
        df_sumET = df_truth_in[[i]]
        df = pd.concat([df_sumET, df0.iloc[:, 1:3]], axis=1) # [phi, eta]
        truth_in_all = [np.array(df)]
        for i in range(1, num_particles):
            df_sumET = df_truth_in[[i]]
            df_sum = dfs_multijet[i-1].iloc[:, 1:3]
            df = pd.concat([df_sumET, df_sum], axis=1)
            truth_in_all += [np.array(df)]
        truth_in_all = np.array(truth_in_all)
        truth_in_all = np.transpose(truth_in_all, axes=[1, 0, 2])
        truth_in = truth_in_all
        in_branch_names = ['sum_ET', 'multijet_phi', 'multijet_eta']
        
    elif model_type == 2:
        truth_in_all = []
        for i in range(0, num_particles):
            df_sumET = df_truth_in[[i]]
            df_sum = dfs_multijet_reverse[i].iloc[:, [3]] # [m]
            df = pd.concat([df_sumET, df_sum], axis=1)
            truth_in_all += [np.array(df)]
        truth_in_all = np.array(truth_in_all)
        truth_in_all = np.transpose(truth_in_all, axes=[1, 0, 2])
        truth_in = truth_in_all    
        in_branch_names = ['sum_ET', 'multijet_m_left']
    
    scale = np.abs(truth_pad).max(axis=0).max(axis=0)
    scale = np.append([thresh_p], scale[1:] + 1e-6)
    scale_in = np.abs(truth_in).max(axis=0).max(axis=0) + 1e-6
    
    truth_pad = truth_pad / scale
    if len(scale_in) > 0:
        truth_in = truth_in / scale_in

    if sample == 'delphes':
        # Split the data into training, testing, and validation
        fold = fold[~drop]
        val = (fold == 0)
        test = (fold == 1)
        train = (fold >= 2)
        val_in, test_in, train_in = truth_in[val][:max_val_evts], truth_in[test][:max_test_evts], truth_in[train][:max_train_evts]
        val_truth_pad, test_truth_pad, train_truth_pad = truth_pad[val][:max_val_evts], truth_pad[test][:max_test_evts], truth_pad[train][:max_train_evts]
        val_mask, test_mask, train_mask = truth_mask[val][:max_val_evts], truth_mask[test][:max_test_evts], truth_mask[train][:max_train_evts]
        
        data = (train_in, train_truth_pad, train_mask, test_in, test_truth_pad, test_mask, val_in, val_truth_pad, val_mask)
        scale = (scale, scale_in)
        label = (in_branch_names, out_branch_names, data_branch_names)
     
    else:
        # Split the data into training and testing    
        num_test_evts = int(truth_pad.shape[0]*test_frac)
        if max_evts and (truth_pad.shape[0] > max_evts):
            num_test_evts = int(max_evts*test_frac)
        if max_test_evts and (num_test_evts > max_test_evts): 
            num_test_evts = max_test_evts
        if max_train_evts and (max_evts > num_test_evts + max_train_evts):
            max_evts = num_test_evts + max_train_evts

        # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        np_rs = RandomState(MT19937(SeedSequence(123456789)))
        idx = np.arange(truth_pad.shape[0])
        np_rs.shuffle(idx)

        truth_pad = truth_pad[idx]
        truth_mask = truth_mask[idx]
        truth_in = truth_in[idx]

        test_in, train_in = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
        test_truth_pad, train_truth_pad = truth_pad[:num_test_evts], truth_pad[num_test_evts:max_evts]
        test_mask, train_mask = truth_mask[:num_test_evts], truth_mask[num_test_evts:max_evts]

        data = (train_in, train_truth_pad, train_mask, test_in, test_truth_pad, test_mask)
        scale = (scale, scale_in)
        label = (in_branch_names, out_branch_names, data_branch_names)
        
    to_return = (data, scale, label)
    return to_return


def pad_array(df, arr, cols):
    m = df[cols[0]].max()
    num_particles = df[cols[0]].values
    if cols[1] in df.columns:
        particles_type = df[cols[1]].values
    else: # placeholder
        particles_type = np.zeros(df.shape[0])
    
    arr_pad = []
    for event, ptype in zip(arr, particles_type):
        x = np.stack(event).T
        x = x[ptype != 4] # exclude MET        
        xpad = np.zeros((m-x.shape[0], x.shape[1]))
        x = np.concatenate((x, xpad))
        arr_pad.append(x)
    arr_pad = np.array(arr_pad)
    
    mask = np.zeros(arr_pad.shape[:2])
    for i in range(mask.shape[0]):
        mask[i, :num_particles[i]] = 1
    mask = mask.astype(bool)    
    return arr_pad, mask
    
    
def convert_label_type(x):
    x = np.floor(x).astype(int)
    x[x < 0] = 0
    x[x >= 5] = 4
    return x


def compute_multijet(x):
    p1 = vector.arr({'pt': x['pt1'], 'phi': x['phi1'], 'eta': x['eta1'], 'mass': x['m1']})
    p2 = vector.arr({'pt': x['pt2'], 'phi': x['phi2'], 'eta': x['eta2'], 'mass': x['m2']})
    p = p1 + p2
    return [p.pt, p.phi, p.eta, p.mass]

    
# Detector effects NF
def read_data_root(file_name, tree_name,
                   out_branch_names, # variables to generate
                   in_branch_names=None, # optional: conditional variables to input
                   data_branch_names=None, # optional: additional variables
                   max_evts=None, max_test_evts=None, testing_frac=0.1,
                   fixed_num_particles=True, num_particles=2, 
                   use_truth_in=False, use_pca=False, use_cond=False, sample='ttH'):
    """
    Read ROOT data file
    Given filename or list of filenames, return train/test inputs and truth
    """
    if type(file_name) != list:
        file_name = [file_name]
    if not in_branch_names:
        in_branch_names = []
    if not data_branch_names:
        data_branch_names = []
    
    dfs = []
    branch_names = in_branch_names + out_branch_names + data_branch_names
    for f in file_name:
        in_file = uproot.open(f)
        tree = in_file[tree_name]
        array_root = tree.arrays(branch_names, library="np")
        df_root = pd.DataFrame(array_root)
        dfs += [df_root]
    df = pd.concat(dfs)

    # get truth variables
    if use_truth_in or use_cond:
        if sample == 'ggH':
            df = df[df['isPassed']].reset_index(drop=True)
            
        if sample == 'delphes':
            df['fold'] = (df['GenMET_phi'] * 1000 % 10).astype(int)
            for i in [0, 1]:
                df['ph_pt'+str(i+1)] = df['smearph_pt'].apply(lambda x: x[i])
                df['ph_phi'+str(i+1)] = df['smearph_phi'].apply(lambda x: x[i])
                df['ph_eta'+str(i+1)] = df['smearph_eta'].apply(lambda x: x[i])

                df['ph_truth_pt'+str(i+1)] = df['genph_pt'].apply(lambda x: x[i])
                df['ph_truth_phi'+str(i+1)] = df['genph_phi'].apply(lambda x: x[i])
                df['ph_truth_eta'+str(i+1)] = df['genph_eta'].apply(lambda x: x[i])
            cols_ordered = ['ph_truth_pt1', 'ph_truth_phi1', 'ph_truth_eta1', 'ph_truth_pt2', 'ph_truth_phi2', 'ph_truth_eta2']
        else:
            df, cols_ordered, cols_added = get_truth_ph(df, sample)
            data_branch_names += cols_added
    
    if use_cond:
        # cols_ordered = ['ph_truth_pt1', 'ph_truth_phi1', 'ph_truth_eta1', 'ph_truth_pt2', 'ph_truth_phi2', 'ph_truth_eta2']
        in_branch_names = cols_ordered + ['mu']
        out_branch_names = ['ph_diff_pt1', 'ph_diff_phi1', 'ph_diff_eta1', 'ph_diff_pt2', 'ph_diff_phi2', 'ph_diff_eta2']
        
        df['ph_diff_pt1'] = df['ph_pt1'] - df['ph_truth_pt1']
        df['ph_diff_phi1'] = df['ph_phi1'] - df['ph_truth_phi1']
        df['ph_diff_eta1'] = df['ph_eta1'] - df['ph_truth_eta1']
        df['ph_diff_pt2'] = df['ph_pt2'] - df['ph_truth_pt2']
        df['ph_diff_phi2'] = df['ph_phi2'] - df['ph_truth_phi2']
        df['ph_diff_eta2'] = df['ph_eta2'] - df['ph_truth_eta2']
        
        x = df['ph_diff_phi1'] > np.pi
        df.loc[x, 'ph_diff_phi1'] = df.loc[x, 'ph_diff_phi1'] - 2*np.pi
        x = df['ph_diff_phi1'] < -np.pi
        df.loc[x, 'ph_diff_phi1'] = df.loc[x, 'ph_diff_phi1'] + 2*np.pi

        x = df['ph_diff_phi2'] > np.pi
        df.loc[x, 'ph_diff_phi2'] = df.loc[x, 'ph_diff_phi2'] - 2*np.pi
        x = df['ph_diff_phi2'] < -np.pi
        df.loc[x, 'ph_diff_phi2'] = df.loc[x, 'ph_diff_phi2'] + 2*np.pi

        if num_particles == 2:
            drop_events = (np.abs(df['ph_diff_pt1']) > 20) | (np.abs(df['ph_diff_pt2']) > 20)
            df = df[~drop_events]
        elif num_particles == 1:
            # reshape to one particle
            cols_ph1 = [b for b in df.columns if '1' in b]
            cols_ph2 = [b for b in df.columns if '2' in b]
            cols = [b[:-1] for b in cols_ph1]
            
            df_ph1 = df[cols_ph1]
            df_ph2 = df[cols_ph2]
            df_ph1.columns = cols
            df_ph2.columns = cols
            df = pd.concat([df_ph1, df_ph2])
            
            drop_events = (np.abs(df['ph_diff_pt']) > 20)
            df = df[~drop_events]
            # eta_cut = (df['ph_truth_eta'].abs() < 1.5) # central
            # eta_cut = (df['ph_truth_eta'].abs() >= 1.5) # forward
            # df = df[eta_cut]

            data_branch_names = [b[:-1] for b in data_branch_names if '1' in b]
            in_branch_names = [b[:-1] for b in in_branch_names if '1' in b]
            out_branch_names = [b[:-1] for b in out_branch_names if '1' in b]
    
    print('\n{} output/truth values, {} input values, {} total data'.format(len(out_branch_names), len(in_branch_names), len(data_branch_names)))
    print('Generating:', out_branch_names)
    print('Input data:', in_branch_names, '\n')
    
    out_particles = df[out_branch_names]
    branch_scale = {}
    
    # transform pt
    if not use_cond:
        for b in out_branch_names:
            if 'pt' in b:
                x = np.log(out_particles[b])
                branch_scale[b] = np.median(x)
                x = x - np.median(x)
                out_particles.loc[:, b] = x
    print("Branch scale:", branch_scale)
        
    # if use_pca:
    #     truth_in = out_particles.to_numpy()
    #     mean_vec = np.mean(truth_in, axis = 0)
    #     truth_in_norm = truth_in - mean_vec
    #     U, S, Vt = np.linalg.svd(truth_in_norm, full_matrices=False)
    #     k = 4
    #     new_basis = Vt[:k].T
    #     out_particles = np.dot(truth_in_norm, new_basis)
    #     out_particles = pd.DataFrame(out_particles, columns = out_branch_names[:k])
    
    if use_cond:
        truth_in_scale = np.array([20, 0.01, 0.005]) # pt, phi, eta
        if num_particles > 1:
            truth_in_scale = np.array(list(truth_in_scale) * num_particles)
        truth_in = out_particles / truth_in_scale
        truth_in = truth_in.to_numpy()
    else:
        # scale to [-1, 1]
        truth_in_scale = out_particles.abs().max() + 1e-6
        truth_in = out_particles / truth_in_scale
        truth_in = truth_in.to_numpy()
        print("Truth scale:", dict(truth_in_scale))
    
    if len(in_branch_names) > 0:
        input_data = df[in_branch_names]
        if not use_cond: # and (not use_pca)
            for b in in_branch_names:
                if 'pt' in b:
                    x = np.log(input_data[b])
                    x = x - np.median(x)
                    input_data.loc[:, b] = x
        
        input_data_scale = input_data.abs().max()
        input_data = input_data / input_data_scale
        input_data = input_data.to_numpy()
    else:
        input_data = np.array([])
        input_data_scale = np.array([])
    print("Input scale:", dict(input_data_scale))
        
    if len(data_branch_names) > 0:
        other_data = df[data_branch_names]
        other_data = other_data.to_numpy()
    else:
        other_data = np.array([])
    
    # for tanh
    drop = (np.abs(truth_in).max(axis=1) >= 1)
    truth_in = truth_in[~drop]
    input_data = input_data[~drop]
    other_data = other_data[~drop]
    
    if sample == 'delphes':
        # Split the data into training, testing, and validation
        fold = df[~drop]['fold'].values      
        val = (fold == 0)
        test = (fold == 1)
        train = (fold >= 2)
        val_in, test_in, train_in = input_data[val], input_data[test], input_data[train]
        val_truth, test_truth, train_truth = truth_in[val], truth_in[test], truth_in[train]
        val_other, test_other, train_other = other_data[val], other_data[test], other_data[train]
           
        print('number of training events:', train_truth.shape)
        print('number of test events:', test_truth.shape)
        print('number of validation events:', val_truth.shape)

        if len(test_in) == 0 and len(train_in) == 0:
            train_in = None
            test_in = None

        data = (train_in, train_truth, train_other, test_in, test_truth, test_other, val_in, val_truth, val_other)
        scale = (branch_scale, truth_in_scale, input_data_scale)
        label = (in_branch_names, out_branch_names, data_branch_names)
    
    else:
        # Split the data into training and testing    
        num_test_evts = int(truth_in.shape[0]*testing_frac)
        if max_evts and (truth_in.shape[0] > max_evts):
            num_test_evts = int(max_evts*testing_frac)
        if max_test_evts and (num_test_evts > max_test_evts): 
            num_test_evts = max_test_evts

        # <NOTE, https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html>
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        np_rs = RandomState(MT19937(SeedSequence(123456789)))
        idx = np.arange(df.shape[0])
        np_rs.shuffle(idx)
        if len(input_data) > 0:
            input_data = input_data[idx]
        if len(truth_in) > 0:
            truth_in = truth_in[idx]
        if len(other_data) > 0:
            other_data = other_data[idx]

        # empty arrays for test_in and train_in if no conditions
        test_in, train_in = input_data[:num_test_evts], input_data[num_test_evts:max_evts]
        test_truth, train_truth = truth_in[:num_test_evts], truth_in[num_test_evts:max_evts]
        test_other, train_other = other_data[:num_test_evts], other_data[num_test_evts:max_evts]

        print('number of training events:', train_truth.shape)
        print('number of test events:', test_truth.shape)

        if len(test_in) == 0 and len(train_in) == 0:
            train_in = None
            test_in = None

        data = (train_in, train_truth, train_other, test_in, test_truth, test_other)
        scale = (branch_scale, truth_in_scale, input_data_scale)
        label = (in_branch_names, out_branch_names, data_branch_names)

    to_return = (data, scale, label)
    # if use_pca:
    #     to_return += (new_basis, mean_vec)
    # else:
    #     to_return += (None, None)
    return to_return


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi


def get_truth_ph(df, sample):
    ### ttH
    def get_ph_idx_ttH(x):
        pid = np.where(x['m_stable_PID'] == 22)[0]
        # pid = pid[np.abs(x['m_stable_eta'][pid]) <= 2.5]
        eid = np.where((x['m_stable_PID'] == 11) | (x['m_stable_PID'] == -11))[0]
        pid_ret = pid[:2]
        if len(pid) > 0:
            p = pid[0]
            eid = eid[eid < p]
            for e in eid:
                # check dR
                delta_eta = x['m_stable_eta'][e] - x['m_stable_eta'][p]
                delta_phi = calc_dphi(x['m_stable_phi'][e], x['m_stable_phi'][p])
                delta_r = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
                if delta_r < 0.1:
                    pid_ret = pid[1:3]
                    break
        if len(pid_ret) < 2:
            pid_ret = np.concatenate([pid_ret, np.repeat(-1, 2-len(pid_ret))])
        else:
            pid_sort = np.argsort(-x['m_stable_pt'][pid_ret])
            pid_ret = pid_ret[pid_sort]
        return pid_ret
    
    ### ggH
    def get_ph_idx_ggH(x):
        num_ph = len(x['m_stable_pt'])
        if num_ph == 0:
            pid_ret = [-1, -1]
        elif num_ph == 1:
            pid_ret = [0, -1]
        else:
            ph_pairs = list(itertools.permutations(range(num_ph), 2))
            dR = []
            dR_pairs = []
            for ph1, ph2 in ph_pairs:
                delta_eta1 = x['ph_eta1'] - x['m_stable_eta'][ph1]
                delta_phi1 = calc_dphi(x['ph_phi1'], x['m_stable_phi'][ph1])
                delta_r1 = np.sqrt(delta_eta1 ** 2 + delta_phi1 ** 2)
                if delta_r1 >= 0.2:
                    continue

                delta_eta2 = x['ph_eta2'] - x['m_stable_eta'][ph2]
                delta_phi2 = calc_dphi(x['ph_phi2'], x['m_stable_phi'][ph2])
                delta_r2 = np.sqrt(delta_eta2 ** 2 + delta_phi2 ** 2)
                if delta_r2 >= 0.2:
                    continue

                dR.append(delta_r1 + delta_r2)
                dR_pairs.append([ph1, ph2])

            if len(dR) == 0:
                pid_ret = [-1, -1]
            else:
                dR_min = np.argmin(dR)
                pid_ret = dR_pairs[dR_min]
        return pid_ret

    if sample == 'ttH':
        x = df.apply(get_ph_idx_ttH, axis = 1)
    elif sample == 'ggH':
        x = df.apply(get_ph_idx_ggH, axis = 1)
    
    ph = pd.DataFrame(x.tolist(), columns = ['ph_truth_idx1', 'ph_truth_idx2'], dtype=int)
    df = pd.concat([df, ph], axis=1)
    # drop events if unable to identify 2 truth photons
    no_truth = (df['ph_truth_idx1'] == -1) | (df['ph_truth_idx2'] == -1)
    df = df[~no_truth]
    
    cols_added = []
    for var in ['pt', 'eta', 'phi', 'm']:
        print(var)
        colname = 'm_stable_'+var
        colnew = 'ph_truth_'+var+'1'
        cols_added.append(colnew)
        df[colnew] = df.apply(lambda x: x[colname][x['ph_truth_idx1']], axis = 1)
        colnew = 'ph_truth_'+var+'2'
        cols_added.append(colnew)
        df[colnew] = df.apply(lambda x: x[colname][x['ph_truth_idx2']], axis = 1)
    
    # exclude m, reorder
    order = [0, 4, 2, 1, 5, 3]
    cols_input = list(np.array(cols_added)[order])
    return df, cols_input, cols_added

def pscale(to_scale, maxi, mini):
    av = (maxi + mini)/2
    hdist = (maxi - mini)/2
    return 0.99999*(to_scale - av)/hdist

def unscale(to_unscale, maxi, mini):
    av = (maxi + mini)/2
    hdist = (maxi - mini)/2
    return to_unscale*hdist/0.99999 + av

import pickle
from sklearn.preprocessing import MinMaxScaler

# <TODO> Use different scaler methods
class InputScaler:
    def __init__(self, feature_range=(-0.99999, 0.99999)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        
    def transform(self, df, outname=None):
        out_df = self.scaler.fit_transform(df)
        if outname is not None:
            self.save(outname)
        return out_df

    def save(self, outname):
        pickle.dump(self.scaler, open(outname, 'wb'))
        return self

    def load(self, outname):
        self.scaler = pickle.load(open(outname, 'rb'))
        return self

    def dump(self):
        print("Min and Max for inputs: {",\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_min_]),\
            ", ".join(["{:.6f}".format(x) for x in self.scaler.data_max_]), "}")


if __name__ == "__main__":
    file_name = "/global/cfs/projectdirs/m3443/data/ForHadronic/train_data/pimode/hadron_pi_mode.root"
    tree_name = "output"
    out_branch_names = ["particle_id", "particle_E", "particle_px", "particle_py", "particle_pz"]
    in_branch_names = ["incoming"]
    data, scale, label, partdict = read_data_root_hadronic(file_name, tree_name, out_branch_names, in_branch_names=in_branch_names, test_frac=0.01, val_frac=0.01)