import pickle
import numpy as np
import cv2

class NoPoseFound(Exception): pass

class PoseEstimator(object):
    def __init__(self, marker_locations, camera_matrix, camera_distortion, min_points=4):
        self.marker_locations = marker_locations
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        self.min_points = min_points

    def __call__(self, markers, irvec=None, itvec=None):
        screens = []
        worlds = []
        seen = set()
        for marker in markers:
            if marker['id'] in seen: continue
            try:
                world = self.marker_locations[marker['id']]
            except KeyError:
                continue
            seen.add(marker['id'])
            screens.extend(marker['verts'])
            worlds.extend(world)
        screens = np.array(screens).astype(np.float32)
        worlds = np.array(worlds).astype(np.float32)
        if len(worlds) < self.min_points:
            raise NoPoseFound("Not enough markers")
        
        useExtrinsicGuess = (irvec is not None and itvec is not None)
        useExtrinsicGuess = False
        if len(worlds) < 8:
            flags=cv2.CV_ITERATIVE
        else:
            flags=cv2.CV_EPNP
        try:
            ret, rvec, tvec = cv2.solvePnP(worlds, screens, self.camera_matrix, self.camera_distortion,
                    rvec=irvec, tvec=itvec, useExtrinsicGuess=useExtrinsicGuess, flags=flags)
            #rvec, tvec, inliers = cv2.solvePnPRansac(worlds, screens, self.camera_matrix, self.camera_distortion)
        except cv2.error, e:
            raise NoPoseFound(e)

        return rvec, tvec

def estimate_poses(marker_data, marker_locations, camera_matrix, camera_distortion, min_points=4):
    #marker_locations = {k: l for k,l in marker_locations.iteritems()
    #    if l['n_observations'] > 100 and l['median_error'] < 0.05}
    estimator = PoseEstimator(marker_locations, camera_matrix, camera_distortion, min_points)
    locations = []
    indices = []
    prvec = None
    ptvec = None
    poses = []
    for i, markers in enumerate(marker_data):
        try:
            rvec, tvec = estimator(markers, irvec=prvec, itvec=ptvec)
            prvec = rvec.copy()
            ptvec = tvec.copy()
        except NoPoseFound:
            continue
        poses.append((i, rvec, tvec))
    return poses
    

def main(camera_spec, marker_locations, marker_file):
    camera = pickle.load(open(camera_spec))
    marker_locations = pickle.load(open(marker_locations))
     
    from pprint import pprint
    marker_data = np.load(marker_file)
    estimate_poses(marker_data, marker_locations, camera['camera_matrix'], camera['dist_coefs'])
    

if __name__ == '__main__':
    import argh
    argh.dispatch_command(main)
