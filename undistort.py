import numpy as np
import cv2
import os.path
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import matplotlib.patches as patches
import sys
from recalibrate import *
from markers import saveMarkers
import av
import Image

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

def gaze_screen_positions(pupil_folder_path, data, start, end):
    pupil_data = load_object(pupil_folder_path + data)

    gaze_list = pupil_data['gaze_positions']
    srf_data_file = load_object(pupil_folder_path + "srf_positions")

    data = get_marker_positions_pixels(srf_data_file)
    
    s = 1080*0.05 + 1080*0.1*0.2
    positions = np.float32([[s, s], [s, 1080 - s], [1920 - s, s], [1920 - s, 1080 - s]]).reshape(-1,1,2)
        
    mark_pos = [[0,0],[0,1],[1,0],[1,1]]

    camera = load_object("camera_calibration")
    cam_dist = camera['dist_coefs']
    cam_dist = np.array([cam_dist[0][0], cam_dist[0][1], 0, 0, 0])
    cam_m = camera['camera_matrix']
    h, w = 720, 1280
    newcamera, roi = cv2.getOptimalNewCameraMatrix(cam_m, cam_dist, (w,h), 0)

    timestamps = np.load(pupil_folder_path + "world_timestamps.npy")
    timestamps_unix = np.load(pupil_folder_path + "world_timestamps_unix.npy")

    cap = cv2.VideoCapture(pupil_folder_path + 'world.mp4')
    
    synchedGaze = correlate_data(gaze_list, timestamps)
    new_gaze_data = []
    #video = cv2.VideoWriter()
    #video.open('video.avi',cv2.cv.CV_FOURCC('M','J','P','G'),30,(1280, 720))
    video = av.open('video.avi', mode = 'w')
    stream = video.add_stream('mpeg4', 30)
    stream.height = 720
    stream.width = 1280

    for i in range(start, end):
        if data[i][1]:
            M = srf_data_file[i]['m_to_screen']

            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLORMAP_AUTUMN)
            
            markers = []
            for pos in mark_pos:
                    pos = np.float32(pos)
                    pos = ref_surface_to_img(pos,M)
                    pos[0]*= 1280
                    pos[1]= 720 - pos[1]*720
                    markers.append(pos)
                        
            markers = np.float32(markers).reshape(-1,1,2)
            markers = cv2.undistortPoints(markers,cam_m,cam_dist, P=newcamera)
            M2, mask = cv2.findHomography(markers, positions, cv2.RANSAC)

            gaze_pts = synchedGaze[i]
            if gaze_pts:
                for g in gaze_pts:
                    gaze = denormalize(g['norm_pos'],(1280, 720),flip_y=True)
                    gaze = np.float32(gaze).reshape(-1,1,2)
                    gaze = cv2.undistortPoints(gaze,cam_m,cam_dist, P=newcamera)
                    screen_pos = tuple(cv2.perspectiveTransform(gaze, M2).reshape(2))
                    new_gaze_data.append((screen_pos[0], screen_pos[1], timestamps_unix[g['index']], g['index'], g['base'][0]['diameter']))

            gray = cv2.undistort(gray, cam_m, cam_dist, None, newcamera)
            gaze = gaze.reshape(2)
            #cv2.circle(gray,tuple(gaze), 1, (255,0,0), thickness=20)
            cv2.circle(gray,tuple(gaze), 1, (0,255,0), thickness=20)


            pil_im = Image.fromarray(gray)
            
            frame = av.VideoFrame.from_image(pil_im)
            packet = stream.encode(frame)
            video.mux(packet)

            cv2.imshow('frame',gray)
            cv2.waitKey(1)
    cap.release()
    video.close()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    path = "/media/usb0/Harkkakurssi/2016_03_09/000/"



    if not os.path.exists(path + "srf_positions"):
        print "Find markers"
        saveMarkers(path)
    if not os.path.exists(path + 'pupil_data_binocular'):
        print "Recalibrate both eyes"
        recalibrate(path)

    start = 60050
    end = 61000
    both = gaze_screen_positions(path, 'pupil_data_binocular', start, end)

    """
    eye0 = gaze_screen_positions(path, 'pupil_data_0')
    eye1 = gaze_screen_positions(path, 'pupil_data_1')



    plt.figure("")
    plt.plot(both[:,2], both[:,0])
    plt.ylim([0,1920])

    plt.figure("both eyes")
    fig = plt.gcf()
    fig.gca().add_patch(patches.Circle((1920/2,1080/2),10, color='black', fill=True))
    plt.hist2d(both[:,0], both[:,1], bins=1000, range = [[0, 1920],[0,1080]])
    plt.ylim([0,1080])
    plt.xlim([0,1920])

    plt.figure("eye 0")
    fig = plt.gcf()
    fig.gca().add_patch(patches.Circle((1920/2,1080/2),10, color='black', fill=True))
    plt.hist2d(eye0[:,0], eye0[:,1], bins=1000, range = [[0, 1920],[0,1080]])
    plt.ylim([0,1080])
    plt.xlim([0,1920])

    plt.figure("eye 1")
    fig = plt.gcf()
    fig.gca().add_patch(patches.Circle((1920/2,1080/2),10, color='black', fill=True))
    plt.hist2d(eye1[:,0], eye1[:,1], bins=1000, range = [[0, 1920],[0,1080]])
    plt.ylim([0,1080])
    plt.xlim([0,1920])

    plt.show()
    """
