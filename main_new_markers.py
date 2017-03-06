import numpy as np
import cv2
import os.path
import math
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches
import sys
#from recalibrate_with_recalculated import *
from markers import saveMarkers
from scipy.interpolate import interp1d
from file_methods import save_object, load_object

def applyProjection(m, vec):
    m = m.reshape(16, 1)
    x = vec[0]
    y = vec[1]
    z = vec[2]
    d = 1 / (m[3]*x + m[7]*y + m[11]*z + m[15])
    x0 = (m[0]*x +m[4]*y +m[8]*z + m[12])*d
    y0 = (m[1]*x +m[5]*y +m[9]*z + m[13])*d
    z0  = (m[2]*x +m[6]*y +m[10]*z + m[14])*d
    return x0, y0, z0

def correlate_data(data,timestamps):
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

    frame_idx = 0
    data_index = 0

    while True:
        try:
            datum = data[data_index]
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
        except IndexError:
            break

        if datum['timestamp'] <= ts:
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1
        else:
            frame_idx+=1
    return data_by_frame

def denormalize(pos, (width, height), flip_y=False):
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y


def ref_surface_to_img(pos,m_to_screen):
    shape = pos.shape
    pos.shape = (-1,1,2)
    new_pos = cv2.perspectiveTransform(pos,m_to_screen )
    new_pos.shape = shape
    return new_pos

def get_marker_positions_pixels(srf_data_file):
    corners = [[0,0],[0,1],[1,1],[1,0]]
    data = []
    for d,i in zip(srf_data_file,range(len(srf_data_file))):
        if d is not None:
            data.append([i,[denormalize(ref_surface_to_img(np.array(c,dtype=np.float32),d['m_to_screen']),(1280,720)) for c in corners]])
        else:
            data.append([i,None])    
    return data

def screen_coords(pos, M):
        pos = np.float32([pos])
        shape = pos.shape
        pos.shape = (-1,1,2)
        new_pos = cv2.perspectiveTransform(pos,M)
        new_pos.shape = shape
        new_pos = new_pos[0]
        new_pos = (new_pos[0], new_pos[1])
        return new_pos

def gaze_screen_positions(pupil_folder_path, data):
    pupil_data = load_object(pupil_folder_path + data)

    gaze_list = pupil_data['gaze_positions']
    timestamps = np.load(pupil_folder_path + "world_timestamps.npy")
    t0 = timestamps[26000]

    for g in gaze_list:
        if g['timestamp'] >= t0:
            g['norm_pos'] = (g['norm_pos'][0], g['norm_pos'][1] + 0.06)

    
    srf_data_file = load_object(pupil_folder_path + "srf_positions")
    
    data = get_marker_positions_pixels(srf_data_file)
    
    x = 1080*0.10 + 1080*0.1*(1/7.)
    y = 1080*0.05 + 1080*0.1*(1/7.)
    y2 = y + 0.1*1080
    l = (5./7.) * 1080*0.1
    xm = 1920/2. - 0.5*l

    
    id32 = [[x,y],[x+l, y], [x+l, y+l], [x, y+l]]
    id4 = [[1920 - x - l,y],[1920 - x, y], [1920 - x, y+l], [1920 - x - l, y+l]]
    id1 = [[1920 - x - l,1080 - y2 - l],[1920 - x, 1080 - y2 - l], [1920 - x, 1080 - y2], [1920 - x - l, 1080 - y2]]
    id5 = [[x,1080 - y2 - l],[x+l, 1080 - y2 - l], [x+l, 1080 - y2], [x, 1080 - y2]]
    id0 = [[xm,1080 - y2 - l],[xm + l, 1080 - y2 - l], [xm + l, 1080 - y2], [xm, 1080 - y2]]
    marker_dict = {"0": id0, "1": id1, "4": id4, "2": id32, "5": id5}

    camera = pickle.load(open("sdcalib.rmap.camera.pickle", 'r'))

    cam_dist = camera['dist_coefs']
    cam_m = camera['camera_matrix']
    h, w = 720, 1280
    #newcamera, roi = cv2.getOptimalNewCameraMatrix(cam_m, cam_dist, (w,h), 0)
    newcamera = pickle.load(open("camera_params_rect.pickle", 'r'))
    newcamera = newcamera['camera_matrix']

    timestamps = np.load(pupil_folder_path + "world_timestamps.npy")
    timestamps_unix = np.load(pupil_folder_path + "world_timestamps_unix.npy")

    unix_ts_interp = interp1d(timestamps, timestamps_unix, axis=0, bounds_error=False)


    data = np.load(path + "markers_new.npy")

    
    synchedGaze = correlate_data(gaze_list, timestamps)
    new_gaze_data = []

    marker_data = {"0": [], "1": [], "2": [], "4": [], "5": []}
    marker_timestamps = {"0": [], "1": [], "2": [], "4": [], "5": []}
    markers_in_frames = []
      
    for i in range(len(data)):
        markers = data[i]
        ts = timestamps[i]
        ts_u = timestamps_unix[i]
        found = []
        for marker in markers:
            if str(marker['id']) in marker_dict.keys():
                found.append(marker['id'])
                marker_verts = marker['verts'].reshape(-1,2)
                marker_data[str(marker['id'])].append(marker_verts)
                marker_timestamps[str(marker['id'])].append(ts)
        markers_in_frames.append(found)


    interpolators = {}
    for key in marker_data.keys():
        marker_interp = interp1d(marker_timestamps[key], marker_data[key], axis=0, bounds_error=False)
        interpolators[key] = marker_interp
    save_object(marker_timestamps, pupil_folder_path + "marker_timestamps")
    save_object(marker_data, pupil_folder_path + "marker_data")

    for gp in synchedGaze:
        for g in gp:
            ts = g['timestamp']
            try:
                if len(g['base']) > 1:
                    ts_u = (g['base'][0]['unix_ts'] + g['base'][1]['unix_ts']) / 2.0
                else:
                    ts_u = g['base'][0]['unix_ts']
            except:
                ts_u = unix_ts_interp(ts)

            index = g['index']
            if len(markers_in_frames[index]) > 2:
                screen_positions = []
                marker_positions = []
                for m in markers_in_frames[index]:
                    marker_verts = interpolators[str(m)](ts)
                    screen_verts = np.float32(marker_dict[str(m)])
                    screen_positions.append(screen_verts)
                    marker_positions.append(marker_verts)

                marker_positions = np.float32(marker_positions).reshape(-1,1,2)
                screen_positions = np.float32(screen_positions).reshape(-1,1,2)

                #marker_positions = cv2.undistortPoints(marker_positions,cam_m,cam_dist, P=newcamera)
                M, mask = cv2.findHomography(marker_positions,screen_positions, 0)

                gaze = denormalize(g['norm_pos'],(1280, 720),flip_y=True)
                gaze = np.float32(gaze).reshape(-1,1,2)
                gaze = cv2.undistortPoints(gaze,cam_m,cam_dist, P=newcamera)
                screen_pos = tuple(cv2.perspectiveTransform(gaze, M).reshape(2))
                new_gaze_data.append((screen_pos[0], screen_pos[1], ts_u, ts, index))
    return np.array(new_gaze_data)


if __name__ == '__main__':

    rootdir = "media/samtuhka/Seagate Expansion Drive/Mittaukset/"
    for dirs in os.walk(rootdir):
        if not dirs[0]==rootdir:
            path = str(dirs[0]) + "/"
            if os.path.exists(path + 'markers_new.npy'):
                print path
                both = gaze_screen_positions(path, 'pupil_data_binocular')
                np.save(path + "screen_coords_binocular.npy", both)
