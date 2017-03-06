import cPickle as pickle
import os
from scipy.interpolate import interp1d
import numpy as np

def load_object(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path,'rb') as fh:
        return pickle.load(fh)

def save_object(object,file_path):
	file_path = os.path.expanduser(file_path)
	with open(file_path,'wb') as fh:
		pickle.dump(object,fh,-1)

#path = "/home/samtuhka/Future16/2016_07_15/000/"
#path = "/home/samtuhka/2016_04_07/000/"

def main(path):
    data_org0 = load_object(path + "pupil_data_0")
    pupil_list0 = data_org0['pupil_positions']
    timestamps0 = [[p['timestamp'], p['unix_ts']] for p in pupil_list0]
    timestamps0 = np.array(timestamps0)

    data_org1 = load_object(path + "pupil_data_1")
    pupil_list1 = data_org1['pupil_positions']
    timestamps1 = [[p['timestamp'], p['unix_ts']] for p in pupil_list1]
    timestamps1 = np.array(timestamps1)


    unix_ts_interp0 = interp1d(timestamps0[:,0], timestamps0[:,1], axis=0, bounds_error=False)
    unix_ts_interp1 = interp1d(timestamps1[:,0], timestamps1[:,1], axis=0, bounds_error=False)


    eye0 = load_object(path + "recalculated_pupil_0")
    eye1 = load_object(path + "recalculated_pupil_1")


    for p in eye0['pupil_positions']:
        p['unix_ts'] = float(unix_ts_interp0(p['timestamp']))
        p['id'] = 0
    for p in eye1['pupil_positions']:
        p['unix_ts'] = float(unix_ts_interp1(p['timestamp']))
        p['id'] = 1

    save_object(eye0,path + "recalculated_pupil_0")
    save_object(eye1,path + "recalculated_pupil_1")
