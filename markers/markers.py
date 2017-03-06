import numpy as np
import cv2
import cPickle as pickle
import inspect, os
import logging

dir_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory

logger = logging.getLogger(__name__)

from square_marker_detect import detect_markers_robust,detect_markers, draw_markers,m_marker_to_screen
from reference_surface import Reference_Surface
from offline_reference_surface import Offline_Reference_Surface
from file_methods import Persistent_Dict,load_object, save_object


def load_surface_definitions_from_file(path, g_pool):
    surface_definitions = Persistent_Dict(os.path.join(dir_path,'surface_definitions'))
    if surface_definitions.get('offline_square_marker_surfaces',[]) != []:
        print("Found ref surfaces defined or copied in previous session.")
        surfaces = [Offline_Reference_Surface(g_pool,saved_definition=d) for d in surface_definitions.get('offline_square_marker_surfaces',[]) if isinstance(d,dict)]
    elif surface_definitions.get('realtime_square_marker_surfaces',[]) != []:
        print("Did not find ref surfaces def created or used by the user in player from earlier session. Loading surfaces defined during capture.")
        surfaces = [Offline_Reference_Surface(g_pool,saved_definition=d) for d in surface_definitions.get('realtime_square_marker_surfaces',[]) if isinstance(d,dict)]
    else:
        print("No surface defs found. Please define using GUI.")
        surfaces = []
    return surfaces

class Global_Container(object):
    pass

def saveMarkers(path):
    cap = cv2.VideoCapture(path + "world.mp4")
    length = float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    m = []
    markers = []

    g_pool = Global_Container()
    surfaces = load_surface_definitions_from_file(path, g_pool)

    camera_calibration = load_object('camera_calibration')
    K = camera_calibration['camera_matrix']
    dist_coefs = camera_calibration['dist_coefs']
    resolution = camera_calibration['resolution']
    camera_intrinsics = K,dist_coefs,resolution
    events = []

    cam_dist = camera_calibration['dist_coefs']
    cam_dist = np.array([cam_dist[0][0], cam_dist[0][1], 0, 0, 0])
    cam_m = camera_calibration['camera_matrix']
    h, w = 720, 1280
    newcamera, roi = cv2.getOptimalNewCameraMatrix(cam_m, cam_dist, (w,h), 0)

    i = 0
    while True:
        ret, frame = cap.read()
        if (type(frame) == type(None)):
                break
        if i%1000 == 0:
            print str((i/length)*100) + "% done"
        i += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.undistort(gray, cam_m, cam_dist, None, newcamera)
        
        m = detect_markers_robust(gray,
                                            grid_size = 5,
                                            prev_markers=m,
                                            min_marker_perimeter=40,
                                            aperture=11,
                                            visualize=0,
                                            true_detect_every_frame=1)
        for s in surfaces:
            s.locate(m, False, camera_intrinsics)
        if s.detected:
            events.append({'m_to_screen':s.m_to_screen,'m_from_screen':s.m_from_screen, 'detected_markers': s.detected_markers})
        else:
            events.append(None)
        markers.append(m)
    np.save(path + "markers_undistorted.npy", markers)
    save_object(events,os.path.join(path ,'srf_positions'))

    cap.release()
    cv2.destroyAllWindows()




