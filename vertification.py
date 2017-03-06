import os
import numpy as np
from file_methods import load_object, save_object
import mapping
import scipy.spatial as sp


def calc_result(pt_cloud):

    pt_cloud[:,0:3:2] *= 1280
    pt_cloud[:,1:4:2] *= 720
    res = np.sqrt(1280**2+720**2)

    field_of_view = 90
    px_per_degree = res/field_of_view

    gaze,ref = pt_cloud[:,0:2],pt_cloud[:,2:4]

    error_lines = np.array([[g,r] for g,r in zip(gaze,ref)])
    error_lines = error_lines.reshape(-1,2)
    error_mag = sp.distance.cdist(gaze,ref).diagonal().copy()
    accuracy_pix = np.mean(error_mag)
    #print("Gaze error mean in world camera pixel: %f"%accuracy_pix)
    error_mag /= px_per_degree
    #print('Error in degrees: %s'%error_mag)
    #print('Outliers: %s'%np.where(error_mag>=5.))
    accuracy = np.mean(error_mag[error_mag<5.])
    print('Angular accuracy: %s'%accuracy)


    #lets calculate precision:  (RMS of distance of succesive samples.)
    # This is a little rough as we do not compensate headmovements in this test.

    # Precision is calculated as the Root Mean Square (RMS)
    # of the angular distance (in degrees of visual angle)
    # between successive samples during a fixation
    #succesive_distances_gaze = sp.distance.cdist(gaze[:-1],gaze[1:]).diagonal().copy()
    #succesive_distances_ref = sp.distance.cdist(ref[:-1],ref[1:]).diagonal().copy()
    #succesive_distances_gaze /=px_per_degree
    #succesive_distances_ref /=px_per_degree
    # if the ref distance is to big we must have moved to a new fixation or there is headmovement,
    # if the gaze dis is to big we can assume human error
    # both times gaze data is not valid for this mesurement
    #succesive_distances =  succesive_distances_gaze[np.logical_and(succesive_distances_gaze< 1., succesive_distances_ref< .1)]
    #precision = np.sqrt(np.mean(succesive_distances**2))

def verification(path):

    timestamps = np.load(path + "world_timestamps_unix.npy")
    timestamps2 = np.load(path + "world_timestamps.npy")
    start = timestamps[0]
    end=timestamps[-1]

    valid_verifications = []
    mappings = []
    rootdir = '/home/samtuhka/pupil/recordings/verifData/'
    for dirs in os.walk(rootdir):
        if not dirs[0]==rootdir:
            ts = float(os.path.basename(dirs[0]))
            if ts >= start and ts <= end:
                valid_verifications.append(dirs[0])


    binocular = load_object(path + "pupil_data_binocular")
    multi = load_object(path + "pupil_data_binocular_multivariate")
    dimensional = load_object(path + "pupil_data_3D")
    left = load_object(path + "pupil_data_0")
    right = load_object(path + "pupil_data_1")
    gaze_datas = [binocular, multi, dimensional, left, right]

    names = ['binocular', 'multivariate', '3D', 'left only', 'right only']
    for data, name in zip(gaze_datas,names):
        gaze_list = data['gaze_positions']
        print name
        for verif in valid_verifications:
            ref_list = list(np.load(verif + "/accuracy_test_ref_list.npy"))
            matched_data = mapping.closest_matches_monocular(gaze_list,ref_list)
            pt_cloud = mapping.preprocess_2d_data_monocular(matched_data)
            pt_cloud = np.array(pt_cloud)
            calc_result(pt_cloud)
        
#path = "/home/samtuhka/2016_04_07/000/"
#verification(path)


