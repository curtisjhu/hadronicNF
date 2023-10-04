import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hickle as hkl

import preprocess

import uproot
import re

from made import create_conditional_flow
from preprocess import pscale, unscale


class Display:
    def __init__(self, model_path, config_path, data_path, max_epochs=1000, epoch=None):
        #LOAD FILE PATHS
        self.model_path = model_path
        self.config_path = config_path
        self.data_path = data_path

        self.max_epochs = max_epochs
        self.epoch = epoch

        outdir = model_path
        self.checkpoint_directory = os.path.join(outdir, 'checkpoints')
        self.log_dir = os.path.join(outdir, 'logs')
        self.csv_dir = os.path.join(outdir, 'csv')
        self.summary_logfile = os.path.join(self.log_dir, 'results.txt')

        #LOAD GENERAL DATA
        self.saved = dict(hkl.load(data_path))
        self.scale = self.saved['scale']
        self.partdict = self.saved['particle_dictionary']
        self.times = None
        self.g_events = None
        self.g_counts = None
        self.g_in = None
        self.g_mask = None

        #LOAD COMPARISON DATA
        self.c_cond_vectors = None
        self.c_counts = None
        self.c_df = None
        self.c_mask = None

        #LOAD MODEL
        config = preprocess.load_yaml(config_path)
        hidden_shape = [config['latent_size']]*config['num_layers']
        self.layers = config['num_bijectors']
        activation = config['activation']
        lr = config['lr']

        dim_truth = 5
        dim_cond = 5
        self.flow_model = create_conditional_flow(hidden_shape, self.layers, 
                                            input_dim=dim_truth, conditional_event_shape=(dim_cond,), out_dim=2,
                                            activation=activation)

        base_lr = lr
        end_lr = 1e-5
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            base_lr, max_epochs, end_lr, power=2)

        # load checkpoint
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
        checkpoint = tf.train.Checkpoint(optimizer=opt, model=self.flow_model)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_directory, max_to_keep=None)
        latest_ckpt = ckpt_manager.latest_checkpoint
        if epoch:
            latest_ckpt = os.path.join(self.checkpoint_directory, 'ckpt-'+str(self.epoch))
        _ = checkpoint.restore(latest_ckpt).expect_partial()
        
        print("Loading latest checkpoint from: {}".format(self.checkpoint_directory))
        if latest_ckpt:
            self.epoch = int(re.findall(r'\/ckpt-(.*)', latest_ckpt)[0]) + 1
            print("Loaded checkpoint from {}".format(latest_ckpt))
        else:
            print("No checkpoints found, please train model first.")



    def load_comparison(self, file_name):
        tree_name = 'output'
        out_branch_names = ["particle_id", "particle_E", "particle_px", "particle_py", "particle_pz"]
        in_branch_names = ['incoming']
        data_branch_names = []

        true_p_counts = np.array([])
        
        if type(file_name) != list:
            file_name = [file_name]

        dfs = []
        branch_names = out_branch_names + in_branch_names + data_branch_names
        for f in file_name:
            in_file = uproot.open(f)
            tree = in_file[tree_name]
            array_root = tree.arrays(branch_names, library="np")
            df_root = pd.DataFrame(array_root)
            dfs += [df_root]
        df = pd.concat(dfs)

        #print(df["incoming"][0])
        #print(df["particle_E"][0])
        cond_vectors = np.zeros(4)

        cond_vectors = np.array([np.asarray(a) for a in df["incoming"].values])
        true_p_counts = np.array([len(a) for a in df["particle_E"]])
        cle = [event[0] for event in df["particle_E"][:]]
        cpx = [event[0] for event in df["particle_px"][:]]
        cpy = [event[0] for event in df["particle_py"][:]]
        cpz = [event[0] for event in df["particle_pz"][:]]
            
        self.c_cond_vectors = cond_vectors[:]
        self.c_counts = true_p_counts
        self.c_df = df
        self.c_mask = [True for i in range(len(true_p_counts))]


    def generate_events(self, num_events, to_folder, silent=False, max_parts=50):
        count = 1
        events_fn = 'events.csv'
        in_fn = 'incoming.npy'
        invpartdict = dict((v, k) for k, v in self.partdict.items())
        in_vectors = self.c_cond_vectors
        if num_events <= np.shape(in_vectors)[0]:
            in_vectors = in_vectors[:num_events]
        elif not silent:
            print("num_events more than length of conditional data, ignored.")
        shift = self.scale[1]*np.ones(np.shape(in_vectors)[0])
        smax = self.scale[0]
        smin = 0
        pcounts = np.array([])

        out_out_vects = []
        out_out_in = []

        index_vect = np.ones((np.shape(in_vectors)[0],1))

        os.makedirs(to_folder, exist_ok=True)
        esave_path = os.path.join(to_folder, events_fn)
        isave_path = os.path.join(to_folder, in_fn)

        with open(esave_path, 'w') as f:
            out_vects = np.zeros((1, np.shape(in_vectors)[0], 5))
            scaled_in_e = pscale(in_vectors[:,0] - shift, smax, smin)
            scaled_in_p = pscale(in_vectors[:,1:], smax, -smax)
            remaining_cond = np.concatenate((scaled_in_e[:, None], scaled_in_p), axis=1)
            index = 1
            while np.shape(remaining_cond)[0] > 0:
                gen_cond = np.hstack((remaining_cond, index*index_vect))
                gen_cond = tf.cast(tf.convert_to_tensor(gen_cond), 'float32')
                cond_kwargs = dict([(f"b{idx}", {"conditional_input": gen_cond}) for idx in range(self.layers)])
                #print("start sample", end="\r", flush=True)
                sample = self.flow_model.sample((np.shape(remaining_cond)[0],), bijector_kwargs=cond_kwargs).numpy()
                #print("end sample", end="\r", flush=True)
                
                partidxs = np.rint(unscale(sample[:,0], np.shape(list(self.partdict.values()))[0], 0))
                partid = np.array([[invpartdict.get(partidx, 0)] for partidx in partidxs])
                #print(partid[:10])
                #print(partid)
                #print(sample[:,1:])
                #print(remaining_cond)
                unscale_r_e = unscale(remaining_cond[:,0], smax, smin)
                unscale_r_p = unscale(remaining_cond[:,1:], smax, -smax)
                unscale_s_e = unscale(sample[:,1], smax, smin)
                unscale_s_p = unscale(sample[:,2:], smax, -smax)

                unscale_r_e = [unscale_r_e] - unscale_s_e
                unscale_r_p = [unscale_r_p] - unscale_s_p

                r_e = pscale(unscale_r_e, smax, smin)[0]
                r_p = pscale(unscale_r_p, smax, -smax)[0]

                remaining_cond = np.concatenate((r_e[:, None], r_p), axis=1)

                # remaining_cond = [unscale(remaining_cond, smax, smin)] - unscale(sample[:,1:], smax, smin)
                # remaining_cond = pscale(remaining_cond, smax, smin)[0]
            
                index = index+1
                out_vects = np.concatenate((out_vects, [np.hstack((partid, np.concatenate((unscale_s_e[:, None], unscale_s_p), axis=1)))]), axis=0)
                #print("out", out_vects[1:, 0, :])  #gets event at index 0
                

                no_more_e = np.array([e < -0.99999 for e in remaining_cond[:,0]])
                if index > max_parts:
                    no_more_e = np.array([True for e in remaining_cond[:,0]])
                #print(out_vects[1:])
                #print("out", out_vects[1:,np.where(no_more_e)[0],:])
                #print(index)
                
                #print(np.shape(out_vects[1:,no_more_e,:]))
                no_more_e_out = out_vects[1:,no_more_e,:]
                out_in = in_vectors[no_more_e,:]
                #print("start for loop", end="\r", flush=True)
                for i in np.arange(np.shape(out_vects[1:,no_more_e,:])[1]):
                    event = no_more_e_out[:,i,:]
                    #print("event", event[:-1])
                    out_out_vects.append(event[:])
                    out_out_in.append(out_in[i,:])
                    #print(out_out_vects)
                    np.savetxt(f, event[:], newline=';   ', footer='\n', comments='')
                    pcounts = np.append(pcounts, np.shape(event)[0])
                    event = np.array([])
                #print("done for loop", end="\r", flush=True) 
                out_vects = out_vects[:,~no_more_e,:]
                if not silent:
                    print(np.shape(out_vects), flush=True)
                remaining_cond = remaining_cond[~no_more_e]
                index_vect = index_vect[~no_more_e]
                in_vectors = in_vectors[~no_more_e]
                
                #pcounts = np.append(pcounts, index)
                if not silent:
                    print(index, end="\r", flush=True)
        f.close()
        np.save(isave_path, out_out_in)
        self.g_events = out_out_vects
        self.g_counts = pcounts
        self.g_in = out_out_in
        self.g_mask = [True for i in range(len(out_out_vects))]
        return out_out_vects, pcounts, out_out_in

    def load_gen(self, from_folder):
        out_vects = []
        i=0
        events_fn = 'events.csv'
        in_fn = 'incoming.npy'
        esave_path = os.path.join(from_folder, events_fn)
        isave_path = os.path.join(from_folder, in_fn)
        with open(esave_path, 'r') as f:
            for line in f:
                event_str = line.split(';   ')
                if i==0:
                    event_str = event_str[:-1]
                    i = 1
                else:
                    event_str = event_str[1:-1]
                event = np.array([[float(v) for v in s.split()] for s in event_str])
                out_vects.append(event)
        out_vects = out_vects[:-1]
        pcounts = np.array([int(np.shape(event)[0]) for event in out_vects])
        g_in = np.load(isave_path)
        self.g_events = out_vects
        self.g_counts = pcounts
        self.g_in = g_in
        self.g_mask = [True for i in range(len(self.g_events))]
        return out_vects, pcounts, g_in


    def make_graphs(self, img_folder, out_out_vects=None, gcounts=None, scale_by_target=False, use_prop=False, particle_num=0, ecut_off=100000):
        if (out_out_vects is None) or (gcounts is None):
            out_out_vects = self.g_events
            gcounts = self.g_counts
            gmask = self.g_mask
        else:
            gmask = [True for i in range(len(out_out_vects))]
        config = dict(histtype='step', lw=2)
        os.makedirs(img_folder, exist_ok=True)
        pn = particle_num

        fontsize=16
        minor_size=14
        leg_size=12

        event_count = len(out_out_vects)
        leading = np.array([event[pn] for m, event in zip(gmask, out_out_vects) if m])
        gcounts = [c for m, c in zip(gmask, gcounts) if m]
        gtype = leading[:,0]
        gidx = np.array([self.partdict.get(id, 0) for id in gtype])
        gle = leading[:,1]
        gpx = leading[:,2]
        gpy = leading[:,3]
        gpz = leading[:,4]
        cmask = [len(event) > pn for event in self.c_df["particle_E"]]
        cmask = [m1 and m2 for m1, m2 in zip(self.c_mask, cmask)]
        ctype = [event[0] for m, event in zip(cmask[:event_count], self.c_df["particle_id"][:event_count]) if m]
        cidx = np.array([self.partdict.get(id, 0) for id in ctype])
        ccounts = [count for m, count in zip(cmask[:event_count], self.c_counts[:event_count]) if m]
        cle = [event[pn] for m, event in zip(cmask[:event_count], self.c_df["particle_E"][:event_count]) if m]
        cpx = [event[pn] for m, event in zip(cmask[:event_count], self.c_df["particle_px"][:event_count]) if m]
        cpy = [event[pn] for m, event in zip(cmask[:event_count], self.c_df["particle_py"][:event_count]) if m]
        cpz = [event[pn] for m, event in zip(cmask[:event_count], self.c_df["particle_pz"][:event_count]) if m]

        if pn == 0:
            graph_labels = ['Leading Particle Energy', 'Leading Particle p_x', 'Leading Particle p_y', 'Leading Particle p_z', 'Incoming 3-p Cross Product Leading 3-p']
            s_graph_labels = ['Outgoing Particle Counts', 'Leading Particle Type']
        elif pn == -1:
            graph_labels = ['Last Particle Energy', 'Last Particle p_x', 'Last Particle p_y', 'Last Particle p_z', 'Incoming 3-p Cross Product Leading 3-p']
            s_graph_labels = ['Outgoing Particle Counts', 'Last Particle Type']
        else:
            graph_labels = [str(pn + 1) + '-th Particle Energy', str(pn + 1) + '-th Particle p_x', str(pn + 1) + '-th Particle p_y', str(pn + 1) + '-th  Particle p_z', 'Incoming 3-p Cross Product Leading 3-p']
            s_graph_labels = ['Outgoing Particle Counts', str(pn + 1) + '-th Particle Type']
        
        graph_list = [[gle, cle], [gpx, cpx], [gpy, cpy], [gpz, cpz]]
        graph_shorthand = ['e', 'px', 'py', 'pz']

        s_graph_list = [[gcounts, ccounts], [gidx, cidx]]
        s_graph_shorthand = ['pcount', 'type']

        if use_prop:
            y_label = 'Proportion of events'
        else:
            y_label = 'Number of events'
    
        fig = plt.figure()
        ax= fig.add_subplot() 

        for i, a in enumerate(s_graph_list):
            save_path = os.path.join(img_folder, s_graph_shorthand[i] + '.png')
            max_v = np.max(a[0])
            max_v = int(max(max_v, np.max(a[1])))
            if use_prop:
                aa, nbins, _ = ax.hist(a[1], bins=max_v + 1, range=[0,max_v + 1], label='Target', weights=np.ones(len(a[1])) / len(a[1]), **config)
                ax.hist(a[0], bins=nbins, range=[0,max_v + 1], label='CNF', weights=np.ones(len(a[0])) / len(a[0]), **config)
            else:
                aa, nbins, _ = ax.hist(a[1], bins=max_v + 1, range=[0,max_v + 1], label='Target', **config)
                ax.hist(a[0], bins=nbins, range=[0,max_v + 1], label='CNF', **config)
            #count_gen, nbins, _ = ax.hist(pcounts, bins=max_count, range=[0,max_count], label='CNF', **config, weights=np.ones(pcounts.shape[0])/pcounts.shape[0])
            #ax.hist(ccounts, bins=nbins, range=[0,max_count], label='Target', **config, weights=np.ones(pcounts.shape[0])/pcounts.shape[0])
            ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
            ax.set_xlabel(s_graph_labels[i], fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.legend(fontsize=leg_size)
            plt.savefig(save_path, bbox_inches='tight')
            ax.clear()

        for i, a in enumerate(graph_list):
            save_path = os.path.join(img_folder, graph_shorthand[i] + '.png')
            max_v = np.max(a[1])
            min_v = np.min(a[1])
            if not scale_by_target:
                max_v = int(max(max_v, np.max(a[0])))
                min_v = int(min(min_v, np.min(a[0])))
            if i==0:
                max_v = min(max_v, ecut_off)
            if use_prop:
                ax.hist(a[1], bins=40, range=[min_v, max_v], weights=np.ones(len(a[1])) / len(a[1]), label='Target', **config)
                ax.hist(a[0], bins=40, range=[min_v, max_v], weights=np.ones(len(a[0])) / len(a[0]), label='CNF', **config)
            else:
                ax.hist(a[1], bins=40, range=[min_v, max_v], label='Target', **config)
                ax.hist(a[0], bins=40, range=[min_v, max_v], label='CNF', **config)
            ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
            ax.set_xlabel(graph_labels[i], fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.legend(fontsize=leg_size)
            plt.savefig(save_path, bbox_inches='tight')
            ax.clear()

        print('Saved to: ', img_folder)

        plt.close()

    def apply_cmask(self, mask):
        if len(mask) == len(self.c_mask):
            self.c_mask = [m1 and m2 for m1, m2 in zip(mask, self.c_mask)]
            print("CMask applied.")
        else:
            print("MASK LENGTH INCORRECT, NOT APPLIED.")
            pass
    
    def apply_gmask(self, mask):
        if len(mask) == len(self.g_mask):
            self.g_mask = [m1 and m2 for m1, m2 in zip(mask, self.g_mask)]
            print("GMask applied.")
        else:
            print("MASK LENGTH INCORRECT, NOT APPLIED.")
            pass
    
    def clear_cmask(self):
        self.c_mask = [True for i in range(len(self.c_counts))]
        print("CMask cleared.")

    def clear_gmask(self):
        self.g_mask = [True for i in range(len(self.g_counts))]
        print("CMask cleared.")

        

    def time_generate_events(self, to_file, runcount, graph_folder, sample_counts=100, mode='total time'):
        fontsize=16
        minor_size=14
        os.makedirs(graph_folder, exist_ok=True)
        num_cond = np.shape(self.c_cond_vectors)[0]
        counts = np.rint(np.linspace(1, num_cond, sample_counts))
        out_vects = []
        for count in counts:
            out_vect = []
            for i in np.arange(runcount):
                start_time = time.time()
                _,_ = self.generate_events(int(count), to_file, silent=True)
                runtime = time.time() - start_time
                if mode=='total time':
                    out_vect.append(runtime)
                elif mode=='per event':
                    out_vect.append(runtime/count)
                elif mode=='throughput':
                    out_vect.append(count/runtime)
                else:
                    print('Mode not recognized, defaulting to total time')
                    out_vect.append(runtime)
                print(runtime, end='\r', flush=True)
            out_vects.append(out_vect)
        times = np.array(out_vects)
        time_av = np.average(times, axis=1)
        fig = plt.figure()
        ax= fig.add_subplot()
        ax.plot(counts, time_av, lw=2)
        ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
        ax.set_xlabel('Number of events generated', fontsize=fontsize)
        if mode=='total time':
            ax.set_ylabel('Time taken to generate (s)', fontsize=fontsize)
            save_path = os.path.join(graph_folder, 'times.png')
        elif mode=='per event':
            ax.set_ylabel('Seconds per event (s/event)', fontsize=fontsize)
            save_path = os.path.join(graph_folder, 'interval.png')
        elif mode=='throughput':
            ax.set_ylabel('Throughput (events/s)', fontsize=fontsize)
            save_path = os.path.join(graph_folder, 'throughput.png')
        else:
            ax.set_ylabel('Time taken to generate (s)', fontsize=fontsize)
            save_path = os.path.join(graph_folder, 'times.png')
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved to: ', save_path)
        ax.clear()
        plt.close()
        self.times = times
        

#EXAMPLE
#MAKE SURE TO CHANGE THE PATHS

if __name__ == "__main__":
    mpath = '/global/homes/a/achen899/normalizing-flow/train_out/pi_mode_20_bijectors'
    cpath = '/global/homes/a/achen899/normalizing-flow/gan4hep/nf/config_nf_hadronic.yml'
    dpath = '/global/cfs/cdirs/m3443/data/ForHadronic/train_data/pimode/pimode.hkl'

    hadron_e_display = Display(mpath, cpath, dpath, max_epochs=1200)
    hadron_e_display.load_comparison("/global/cfs/projectdirs/m3443/data/ForHadronic/train_data/pimode/hadron_pi_mode.root")
    print("Comparison data loaded.")
    events, counts, g_in = hadron_e_display.generate_events(100000, "out_nf_pi_20_b_163", max_parts=40)
    print("Generated: ", events[0])
    hadron_e_display.make_graphs("/global/homes/a/achen899/normalizing-flow/gan4hep/nf/hadronic_graph", events, counts, scale_by_target=True)
    c_mask = [c > 15 for c in hadron_e_display.c_counts]
    hadron_e_display.apply_cmask(c_mask)
    g_mask = [c > 15 for c in counts]
    hadron_e_display.apply_gmask(g_mask)

    hadron_e_display.make_graphs("/global/homes/a/achen899/normalizing-flow/gan4hep/nf/hadronic_graph_inv/pi_mode_20_b/163/test", particle_num=4, ecut_off = 10000)
