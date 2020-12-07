import json
import os
from os import times

import numpy as np
from scipy import interpolate


SEQ_LENGTH = 200
DATA_PATH = ".\\data"
DATASETS = ["1", "2", "3", "4", "A", "B", "C", "D"]

NOISE_LEVEL = 1e-3
MAX_CUT_PERCENTAGE = 0.05
SAMPLE_TO_EXPAND = 1


def add_noise_3d(seq):
    seq = np.array(seq)
    noise = np.random.random([seq.shape[0], 3]) * NOISE_LEVEL
    seq += noise
    return seq.tolist()


def random_cut(speed_accel, angle_accel, timestamp):
    n_original = len(timestamp)
    cut_front = round(np.random.random(1)[0] * MAX_CUT_PERCENTAGE * n_original)
    cut_back = n_original - round(np.random.random(1)[0] * MAX_CUT_PERCENTAGE * n_original)
    return speed_accel[cut_front: cut_back], angle_accel[cut_front: cut_back], timestamp[cut_front: cut_back]


def normalize3d(seq):
    norms = np.linalg.norm(seq, axis=1)
    maxval = np.max(norms)
    return seq / maxval


def uniform_timestamp(ts):
    ts_arr = np.array(ts)
    return (ts_arr - ts_arr[0]).tolist()


def uniform_accel(accel, ts):
    ts_old = np.array(ts)
    accel_arr = np.array(accel)
    interp_x = interpolate.interp1d(ts_old, accel_arr[:,0], kind='slinear')
    interp_y = interpolate.interp1d(ts_old, accel_arr[:,1], kind='slinear')
    interp_z = interpolate.interp1d(ts_old, accel_arr[:,2], kind='slinear')
    ts_new = np.linspace(0, ts_old[-1], SEQ_LENGTH)
    accel_new_x = interp_x(ts_new)
    accel_new_y = interp_y(ts_new)
    accel_new_z = interp_z(ts_new)
    accel_new = normalize3d(np.vstack((accel_new_x, accel_new_y, accel_new_z)).T)
    return accel_new


if __name__ == "__main__":
    for ds_name in DATASETS:
        ds_path = os.path.join(DATA_PATH, ds_name+".json")
        with open(ds_path, "r") as dsfp:
            ds = json.load(dsfp)
        sequences = []
        for fn in ds.keys():
            timestamp = ds[fn]["timestamp"]
            speed_accel = ds[fn]["speed_accel"]
            angle_accel = ds[fn]["angle_accel"]
            # for each sample, expand it to SAMPLE_TO_EXPAND samples
            samples = 0
            while samples < SAMPLE_TO_EXPAND:
                # uniform
                timestamp = uniform_timestamp(timestamp)
                speed_accel = uniform_accel(speed_accel, timestamp)
                angle_accel = uniform_accel(angle_accel, timestamp)
                seq = np.hstack((speed_accel, angle_accel))
                # add to sequence
                sequences.append(seq.tolist())
                samples += 1
                # randomize
                speed_accel, angle_accel, timestamp = random_cut(ds[fn]["speed_accel"], ds[fn]["angle_accel"], ds[fn]["timestamp"])
                speed_accel = add_noise_3d(speed_accel)
                angle_accel = add_noise_3d(angle_accel)
        # save
        uniformed_path = os.path.join(DATA_PATH, ds_name+"_uniformed.json")
        with open(uniformed_path, "w") as unifp:
            json.dump(sequences, unifp)
            print("dumped {}".format(unifp.name))
    
