import os
import numpy as np
from file_methods import load_object, save_object
import mapping
import scipy.spatial as sp
from bisect import bisect_left



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


    gaze_h = gaze[:,0]
    ref_h = ref[:,0]
    error_h = np.abs(gaze_h - ref_h)
    error_h /= px_per_degree
    accuracy_h = np.mean(error_h[error_mag<5.])

    gaze_v = gaze[:,1]
    ref_v = ref[:,1]
    error_v = np.abs(gaze_v - ref_v)
    error_v /= px_per_degree
    accuracy_v = np.mean(error_v[error_mag<5.])
    print('Angular accuracy: %s (hor: %s, ver: %s)' %(accuracy, accuracy_h, accuracy_v))


    error_h_alt = (gaze_h - ref_h)
    error_h_alt /= px_per_degree
    accuracy_h_alt = np.mean(error_h_alt[error_mag<5.])

    error_v_alt = (gaze_v - ref_v)
    error_v_alt /= px_per_degree
    accuracy_v_alt = np.mean(error_v_alt[error_mag<5.])


    return accuracy, accuracy_h, accuracy_v, gaze_h, ref_h, gaze_v, ref_v, accuracy_h_alt, accuracy_v_alt
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


def closest(a, x):
    i = bisect_left(a, x)
    if i >= 0 and i != len(a):
        d = abs(x - a[i])
        if d < 1/40.:
            return True
    return False

def verification(path):

    timestamps = np.load(path + "/world_timestamps_unix.npy")
    timestamps2 = np.load(path + "/world_timestamps.npy")
    start = timestamps[0]
    end=timestamps[-1]

    valid_verifications = []
    mappings = []
    rootdir = '/home/samtuhka/pupil/recordings/verifData/'
    rootdir = '/media/samtuhka/Seagate Expansion Drive/Mittaukset/verifData'
    for dirs in os.walk(rootdir):
        if not dirs[0]==rootdir:
            ts = float(os.path.basename(dirs[0]))
            if ts >= start and ts <= end:
                valid_verifications.append(dirs[0])

    valid_verifications = sorted(valid_verifications, key=lambda k:  float(os.path.basename(k)))

    #try:
    #    binocular = load_object(path + "/pupil_data_binocular_repaired")
    #except:
    #    binocular = load_object(path + "/pupil_data_binocular")
    binocular = load_object(path + "/pupil_data_binocular")
    #multi = load_object(path + "/pupil_data_binocular_multivariate")
    #dimensional = load_object(path + "/pupil_data_3D")
    #left = load_object(path + "/pupil_data_0")
    #right = load_object(path + "/pupil_data_1")

    #dic = {'binocular':binocular, 'multivariate':multi, '3D':dimensional, 'left only':left, 'right only':right}
    dic = {'binocular':binocular}

    refs = {}

    bin_timestamps = [g['timestamp'] for g in binocular['gaze_positions']]

    
    for key in dic.keys():
        data = dic[key]
        gaze_list = data['gaze_positions']
        gaze_list = [g for g in gaze_list if closest(bin_timestamps, g['timestamp'])]
        dic[key] = []
        refs[key] = {'hor':[], 'ver': [], 'hor_ref': [], 'ver_ref': []}
        print key
        for verif in valid_verifications:
            ref_list = list(np.load(verif + "/accuracy_test_ref_list.npy"))
            ref_list = [r for r in ref_list if r['screenpos'][0] > 300 and r['screenpos'][0] < 1920 - 300 and r['screenpos'][1] > 200 and r['screenpos'][1] < 880]
            matched_data = mapping.closest_matches_monocular(gaze_list,ref_list)
            
            pt_cloud = mapping.preprocess_2d_data_monocular(matched_data)
            pt_cloud = np.array(pt_cloud)
            try:
                acc, acc_h, acc_v, gaze_h, ref_h, gaze_v, ref_v, accuracy_h_alt, accuracy_v_alt = calc_result(pt_cloud)
                refs[key]['hor'].append(gaze_h)
                refs[key]['ver'].append(gaze_v)
                refs[key]['hor_ref'].append(ref_h)
                refs[key]['ver_ref'].append(ref_v)
                dic[key].append([acc, acc_h, acc_v, accuracy_h_alt, accuracy_v_alt])
            except:
                dic[key].append([-1, -1, -1, -1, -1])   
    save_object(dic, path + "/verif_results_sim")
    save_object(refs, path + "/verif_data_sim")        
        
#path = "/home/samtuhka/2016_04_07/000/"
#verification(path)



    
rootdir = path = "/media/samtuhka/Seagate Expansion Drive/Mittaukset/"
#purged_datas = ["2016_04_08/001", "2016_04_12/001", "2016_04_19/000"]
purged_datas = []
for dirs in os.walk(rootdir):
    path = str(dirs[0]) + "/"
    skip = False
    for p in purged_datas:
            p = rootdir + p
            p = os.path.normpath(p)
            if p in os.path.normpath(path):
                skip = True
    if os.path.exists(path + "screen_coords_binocular.npy") and  os.path.exists(path + "sim_data.msgpack") and not skip:
        print path
        verification(path)


