import numpy as np
from pandas import read_csv
from tensorflow import ragged as tfr
from typing import Tuple, List
import pprint
import uproot
import awkward as ak
import os
#Converts .csv at filename to .root at out_file
        

GAN_INPUT_DATA_TYPE = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]

#num of events, event shape 
#num of events, 1
def convert(filename, out_file):
    
    d = read_csv(filename, header=None)

    df = d.values

    in_data = []
    truth_data = []
    count = 1
    for d_string_list in df:
        arr_float = np.array([])
        for d_string in np.char.split(d_string_list[0], sep=';').tolist():
            if arr_float.size == 0:
                arr_float = np.array([np.fromstring(d_string, sep = ' ')])
            else: 
                arr_float = np.concatenate((arr_float, np.array([np.fromstring(d_string, sep = ' ')])), axis=0)
        truth_data_n = arr_float[1:]
        #truth_data_n = np.float32(np.array(np.flipud(sorted(truth_data_n, key=lambda x:x[1]))))
        truth_data_n = np.float32(np.array(truth_data_n))
        #truth_data_n = scaler.transform(truth_data_n)
        in_data.append(np.float32(arr_float[0][1:]))
        truth_data.append(truth_data_n)
        if count % 1000== 0:
            print(count, end="\r", flush=True)
        count+=1

    #num_test_evts = int(len(in_data)*test_frac)
    #if num_test_evts > 10_000: num_test_evts = 10_000
    #print("WARNING: num_test_evts decreased to 10,000 automatically")

    #in_data = scaler.transform(in_data.reshape(-1,1))
    
    out_branch_names = ['particle_id', "particle_E", "particle_px", "particle_py", "particle_pz"]
    in_branch_names = ['incoming']
    branch_names = out_branch_names + in_branch_names
    
    if (os.path.isfile(out_file)):
        f = uproot.open(out_file)
        tree = f["output"]
        all_dict = tree.arrays(branch_names, library="np")
        exists = True
    else:
        f = uproot.create(out_file)
        all_dict = {}
        exists = False
    print(exists)
    count = 0
    for out_label in out_branch_names:
        truth_data_n = [arr[:, count] for arr in truth_data]
        if exists:
            current = [np.float32(arr) for arr in all_dict[out_label]]
            all_dict[out_label] = current + truth_data_n
        else:
            all_dict[out_label] = truth_data_n
        count+=1
    if exists:
        current = [np.float32(arr) for arr in all_dict[in_branch_names[0]]]
        all_dict[in_branch_names[0]] = current + in_data
    else:
        all_dict[in_branch_names[0]] = in_data
    #pprint.pprint(all_dict)

    for key in all_dict:
        all_dict[key] = ak.Array(all_dict[key])
    os.remove(out_file)
  
    print(len(all_dict['incoming']))
    print(len(all_dict['particle_id']))
    b = uproot.create(out_file)
    b["output"] = all_dict