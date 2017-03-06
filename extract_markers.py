#!/usr/bin/env python2

import sys
import cv2
import argh
import pickle
import square_marker_detect as markerdetect
import numpy as np


def marker_positions(camera_spec, videofile, outfile, new_camera=None, start_time=0.0, end_time=float("inf"), visualize=False,
        output_camera=None):
    camera = pickle.load(open(camera_spec, 'r'))
    image_resolution = camera['resolution']
    
    if 'rect_map' not in camera:
        camera_matrix = camera['camera_matrix']
        camera_distortion = camera['dist_coefs']
        rect_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_resolution, 0.0)
        rmap = cv2.initUndistortRectifyMap(
            camera_matrix, camera_distortion, None, rect_camera_matrix, image_resolution,
            cv2.CV_32FC1)
    else:
        rmap = camera['rect_map']
        rect_camera_matrix = camera['rect_camera_matrix']

    
    camera = {}
    camera['camera_matrix'] = rect_camera_matrix
    camera['dist_coefs'] = None
    camera['resolution'] = image_resolution
    if new_camera is not None:
        pickle.dump(camera, open(new_camera, 'w'))

   
    video = cv2.VideoCapture(videofile)
    video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time*1000)
    frames = []
    while True:
        ret, oframe = video.read()
        frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        msecs = video.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
        time = msecs/1000.0
        if time > end_time:
            break
        markers = markerdetect.detect_markers(frame, 5)
        frame_d = {
                'ts': time,
                'markers': markers,
                }
        frames.append(frame_d)
        
        if not visualize: continue
        markerdetect.draw_markers(frame, frame_d['markers'])
        #for marker in markers:
        #    for i, corner in enumerate(marker['verts']):
        #        cv2.putText(frame, str(i), tuple(np.int0(corner[0,:])),
        #                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,100,50))
        #print markers
        cv2.imshow('video', frame)
        cv2.waitKey(1)
    np.save(outfile, frames)


if __name__ == '__main__':
    argh.dispatch_command(marker_positions)
