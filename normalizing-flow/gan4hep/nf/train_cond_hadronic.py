"""
Trainer for Normalizing Flow
"""
import os
import time
import re

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
import hickle as hkl

from utils import train_density_estimation_cond
from utils_plot_recurrent import *

from preprocess import pscale, unscale

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

    
def evaluate(flow_model, testing_data, cond_kwargs):
    num_samples, num_dims = testing_data.shape
    samples = flow_model.sample(num_samples, bijector_kwargs=cond_kwargs).numpy()
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]
    return np.average(distances), samples

def evaluate_b(flow_model, testing_data, in_parts, partdict, scale):
    pcounts = np.array([])
    
    smax = scale[0]
    smin = 0
    #in_vectors = unscale(in_parts[:,:4], smax, smin)
    in_vectors = in_parts[:, :4]
    invpartdict = dict((v, k) for k, v in partdict.items())
    num_samples, num_dims = testing_data.shape
    out_vects = np.zeros((1, np.shape(in_vectors)[0], 5))
    
    # scaled_in_e = pscale(in_vectors[:,0], smax, smin)
    # scaled_in_p = pscale(in_vectors[:,1:], smax, -smax)
    # remaining_cond = np.concatenate((scaled_in_e[:, None], scaled_in_p), axis=1)
    # remaining_cond = pscale(in_vectors, scale[0], scale[1])
    remaining_cond = in_vectors
    
    index = 1
    index_vect = np.ones((np.shape(in_vectors)[0],1))
    out_out_vects = []
    while np.shape(remaining_cond)[0] > 0:
        gen_cond = np.hstack((remaining_cond, index*index_vect))
        gen_cond = tf.cast(tf.convert_to_tensor(gen_cond), 'float32')
        cond_kwargs = dict([(f"b{idx}", {"conditional_input": gen_cond}) for idx in range(layers)])
        #print("start sample", end="\r", flush=True)
        sample = flow_model.sample((np.shape(remaining_cond)[0],), bijector_kwargs=cond_kwargs).numpy()
        #print("end sample", end="\r", flush=True)

        partidxs = np.rint(unscale(sample[:,0], np.shape(list(partdict.values()))[0], 0))
        partid = np.array([[invpartdict.get(partidx, 0)] for partidx in partidxs])

        sample = np.float64(sample)

        unscale_r_e = unscale(remaining_cond[:,0], smax, smin)
        unscale_r_p = unscale(remaining_cond[:,1:], smax, -smax)
        unscale_s_e = unscale(sample[:,1], smax, smin)
        unscale_s_p = unscale(sample[:,2:], smax, -smax)

        unscale_r_e = [unscale_r_e] - unscale_s_e
        unscale_r_p = [unscale_r_p] - unscale_s_p

        r_e = pscale(unscale_r_e, smax, smin)[0]
        r_p = pscale(unscale_r_p, smax, -smax)[0]

        remaining_cond = np.concatenate((r_e[:, None], r_p), axis=1)
        #print("sample: ", partid[0], unscale_s_e[0], unscale_s_p[0])

        index = index+1
        out_vects = np.concatenate((out_vects, [np.hstack((pscale(np.array([partidxs]).T, np.shape(list(partdict.values()))[0], 0), sample[:,1:]))]), axis=0)

        no_more_e = np.array([e < -0.99999 for e in remaining_cond[:,0]])

        no_more_e_out = out_vects[1:,no_more_e,:]
        if (np.shape(out_vects)[0] > 500):
            for i in np.arange(np.shape(out_vects[1:,:,:])[1]):
                event = out_vects[:,i,:]
                out_out_vects.append(event[:])
                pcounts = np.append(pcounts, 100*np.shape(event)[0] - 1)
                event = np.array([])
            out_vects = np.array([])
            remaining_cond = np.array([])
        else:
            for i in np.arange(np.shape(out_vects[1:,no_more_e,:])[1]):
                event = no_more_e_out[:,i,:]
                out_out_vects.append(event[:])
                if (np.shape(out_vects)[0] > 100):
                    pcounts = np.append(pcounts, 100*np.shape(event)[0] - 1)
                else:
                    pcounts = np.append(pcounts, np.shape(event)[0] - 1)
                event = np.array([])

            out_vects = out_vects[:,~no_more_e,:]
            remaining_cond = remaining_cond[~no_more_e]

        #print(np.shape(out_vects), end="\r", flush=True)      
        index_vect = index_vect[~no_more_e]
        

    total_parts = np.sum(pcounts)
    #print(total_parts)
    #print(np.shape(testing_data)[0])
    samples = np.concatenate(out_out_vects)
    
    part_num_disp = max((total_parts - np.shape(testing_data)[0])/np.shape(testing_data)[0], (np.shape(testing_data)[0] - total_parts)/total_parts)
    #print(part_num_disp)
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]
    distances.append(part_num_disp)
    return np.average(distances), samples

def train(train_truth, train_in, testing_truth, testing_in, flow_model, lr, batch_size, layers, max_epochs, outdir, plot_config, partdict, scale, val_truth=None, val_in=None):
    use_val = True
    if val_in is None:
        print("not using val")
        use_val = False
    base_lr = lr
    end_lr = 1e-5
    nbatches = train_truth.shape[0] // batch_size + 1
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs * nbatches, end_lr, power=2)

    labels = plot_config['labels']
    labels_norm = ['log '+labels[0]] + list(labels[1:])
    
    log_dir = os.path.join(outdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_dir = os.path.join(outdir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    img_dir = os.path.join(outdir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    
    # initialize checkpoints
    checkpoint_directory = os.path.join(outdir, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    latest_ckpt = ckpt_manager.latest_checkpoint
    _ = checkpoint.restore(latest_ckpt).expect_partial()
    
    print("Loading latest checkpoint from: {}".format(checkpoint_directory))
    if latest_ckpt:
        start_epoch = int(re.findall(r'\/ckpt-(.*)', latest_ckpt)[0]) + 1
        print("Restored from {}".format(latest_ckpt))
    else:
        start_epoch = 0
        print("Initializing from scratch.")

    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        (train_in, train_truth))
    if training_data.cardinality() < 1000:
        bsize = 1000
    else:
        bsize = 1000 + np.rint(training_data.cardinality()/1000)
    training_data = training_data.shuffle(buffer_size=bsize)
    training_data = training_data.batch(batch_size).prefetch(AUTO)

    # start training
    summary_logfile = os.path.join(log_dir, 'results.txt')
    if use_val:
        tmp_res = "# Epoch, Time, WD (Wasserstein distance), Ltr (training loss), Lte (testing loss), Lva (validation loss)" 
    else:
        tmp_res = "# Epoch, Time, WD (Wasserstein distance), Ltr (training loss), Lte (testing loss)" 
    with open(summary_logfile, 'a') as f:
        f.write(tmp_res + "\n")
    # save learning rate
    lr_logfile = os.path.join(log_dir, 'learning_rate.txt')
    tmp_res = "# Epoch, " + ", ".join(map(str, range(nbatches)))
    with open(lr_logfile, 'a') as f:
        f.write(tmp_res + "\n")
    print("idx, train loss, distance, minimum distance, minimum epoch")
    
    # Get min wass dist so far
    df = pd.read_csv(summary_logfile)
    if use_val:
        df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te', 'loss_va']
    else:
        df.columns = ['epoch', 'time', 'wass_dist', 'loss_tr', 'loss_te']
    
    df = df[df['epoch'].str.contains(r'\*')]
    if len(df) == 0:
        min_wdis, min_iepoch = 9999, -1
    else:
        df['epoch'] = df['epoch'].apply(lambda x: x[2:])
        df = df.reset_index(drop = True).astype('float64')
        min_wdis, min_iepoch = df[['wass_dist', 'epoch']].sort_values('wass_dist').iloc[0]
        min_iepoch = int(min_iepoch)

    delta_stop = 1000
    start_time = time.time()
    testing_batch = tf.cast(tf.convert_to_tensor(testing_truth), 'float32')
    testing_cond = tf.cast(tf.convert_to_tensor(testing_in), 'float32')
    cond_kwargs = dict([(f"b{idx}", {"conditional_input": testing_cond}) for idx in range(layers)])
    if use_val:
        validation_batch = tf.cast(tf.convert_to_tensor(val_truth), 'float32')
        validation_cond = tf.cast(tf.convert_to_tensor(val_in), 'float32')
        vcond_kwargs = dict([(f"b{idx}", {"conditional_input": validation_cond}) for idx in range(layers)])

    start_mask = np.array([(in_idx[4] == 1) for in_idx in testing_in])
    in_parts = test_in[start_mask]

    
    
    for i in range(start_epoch, max_epochs):
        lr = []
        train_loss = []
        for batch in training_data:
            condition, batch = batch
            batch = tf.cast(batch, 'float32')
            condition = tf.cast(condition, 'float32')
            # tf.debugging.check_numerics(batch, "batch HAS NAN")
            # tf.debugging.check_numerics(condition, "cond HAS NAN")
            train_loss += [train_density_estimation_cond(flow_model, opt, batch, condition, layers)]

            # tf.debugging.check_numerics(train_loss, "train loss HAS NAN")
            if isinstance(opt.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr += [opt.lr(opt.iterations).numpy()]
            else:
                lr += [opt.lr.numpy()]
                
        train_loss = np.array(train_loss)
        avg_loss = np.sum(train_loss, axis=0)/train_loss.shape[0]
        elapsed = time.time() - start_time
        
        wdis, predictions = evaluate(flow_model, testing_truth, cond_kwargs)

        #wdis, predictions = evaluate_b(flow_model, testing_truth, in_parts, partdict, scale)
        test_loss = -tf.reduce_mean(flow_model.log_prob(testing_batch, bijector_kwargs=cond_kwargs))
        if use_val:
            val_loss = -tf.reduce_mean(flow_model.log_prob(validation_batch, bijector_kwargs=vcond_kwargs))
        
        if wdis < min_wdis:
            compare(predictions, testing_truth, labels_norm, 1, img_dir, i)
            min_wdis = wdis
            min_iepoch = i
            save_path = os.path.join(csv_dir, 'pred_epoch_{:04d}.csv'.format(i))
            #pd.DataFrame(predictions, columns = labels).to_csv(save_path, index=False)
        elif i - min_iepoch > delta_stop:
            plot_logfile(summary_logfile, i, img_dir, use_val = use_val)
            break
        ckpt_manager.save(checkpoint_number = i)
        
        if (i % 50 == 0) or (i == max_epochs - 1):
            plot_logfile(summary_logfile, i, img_dir, lr_filename=lr_logfile, use_val = use_val)
        
        tmp_res = "* {:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f}".format(i, elapsed, wdis, avg_loss, test_loss)
        if use_val:
            tmp_res = "* {:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(i, elapsed, wdis, avg_loss, test_loss, val_loss)
        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")
        tmp_res = ", ".join(map(str, lr))
        with open(lr_logfile, 'a') as f:
            f.write("* {:05d}, ".format(i) + tmp_res + "\n")
        print(f"{i}, {train_loss[-1]:.4f}, {wdis:.4f}, {min_wdis:.4f}, {min_iepoch}", flush=True)
        print('Epoch complete at ', time.time() - start_time)
        

#python train_cond_hadronic.py --config_file "/global/homes/a/achen899/normalizing-flow/gan4hep/nf/config_nf_hadronic.yml" --log-dir "/global/homes/a/achen899/normalizing-flow/train_out/hadronic_test" --epochs 50


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg("--config_file", help='configuration file')
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--data-dir", help='data directory')
    add_arg("--epochs", help='number of maximum epochs', default=500, type=int)
    args = parser.parse_args()

    import preprocess
    from made import create_conditional_flow

    # =============================
    # load preprocessed data
    # =============================
    config = preprocess.load_yaml(args.config_file)
    file_name = config['file_name']
    tree_name = config['tree_name']
    out_branch_names = config['out_branch_names']
    truth_branch_names = config['truth_branch_names']
    data_branch_names = config['data_branch_names']
    in_branch_names = config['in_branch_names']

    outdir = os.path.join('trained_results', args.log_dir)
    save_path = args.data_dir
    if os.path.exists(save_path):
        saved = dict(hkl.load(save_path))
        data = saved['data']
        scale = saved['scale']
        label = saved['label']
        partdict = saved['particle_dictionary']
    else:
        data, scale, label, partdict = preprocess.read_data_root_hadronic(file_name, tree_name, out_branch_names, 
                                                                 in_branch_names=in_branch_names, test_frac=0.01, val_frac=0.01)
        to_save = {'data':data, 'scale':scale, 'label':label, 'particle_dictionary':partdict}
        os.makedirs(outdir, exist_ok=True)
        hkl.dump(to_save, save_path)
        print('Saved:', save_path)
        
    if len(data) == 4: # no validation set
        train_in, train_truth, test_in, test_truth = data
        val_in = None
        val_truth = None
        print("in", np.max(train_in))
        print("truth", np.max(train_truth))
    else:
        train_in, train_truth, test_in, test_truth, val_in, val_truth = data
        print("Using validation set.")
        print("in", np.max(train_in))
        print("truth", np.max(train_truth))
    in_branch_names, out_branch_names, data_branch_names = label
    scale = scale

    print()
    print('Generating:', out_branch_names)
    print('Input conditions:', in_branch_names)
    print('training:', train_in.shape, train_truth.shape)
    print('test:', test_in.shape, test_truth.shape) 
    #print('test:', val_in.shape, val_truth.shape)
    print()
    
    hidden_shape = [config['latent_size']]*config['num_layers']
    layers = config['num_bijectors']
    activation = config['activation']
    lr = config['lr']
    batch_size = config['batch_size']
    max_epochs = args.epochs
    
    dim_truth = train_truth.shape[1]
    print(dim_truth)
    dim_cond = train_in.shape[1]
    print(dim_cond)


    #print(train_in)
    #print(train_truth)
    
    maf = create_conditional_flow(hidden_shape, layers, 
                                  input_dim=dim_truth, conditional_event_shape=(dim_cond-1,), out_dim=2,
                                  activation=activation)
    
    plot_config = {'labels': out_branch_names}
    train_truth=train_truth[:,1:]
    test_truth=test_truth[:,1:]
    val_truth=val_truth[:,1:]
    train(train_truth, train_in, test_truth, test_in, maf, lr, batch_size, layers, max_epochs, outdir, plot_config, partdict, scale, val_truth=val_truth, val_in=val_in)
