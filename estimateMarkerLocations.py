import numpy as np
import cv2
from collections import defaultdict

def get_frame_locations(markers, marker_coords, reference_id):
    for marker in markers:
        marker_id = marker['id']
        marker_world_points = marker_coords(marker_id)
        marker_screen_points = marker['verts']
        retval, rvec, tvec = cv2.solvePnP(marker_world_points, marker_positions, cam_m, cam_dist)

import subprocess
def planarPnP(world_positions, screen_positions, camera_matrix, camera_distortion):
    screen_positions = cv2.undistortPoints(screen_positions, camera_matrix, camera_distortion)
    proc = subprocess.Popen(['./csvpose'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stuff = ""
    for s, w in zip(screen_positions.reshape(-1, 2), world_positions):
        stuff += "%f %f 1 %f %f %f\n"%tuple(list(s) + list(w))
    stuff = proc.communicate(stuff)
    stuff = stuff[0].split("\n")
    tvec = np.array(map(float, stuff[0].split()))
    rvec = np.array(map(float, stuff[1].split()))
    return tvec, rvec

class MarkerGeometryEstimator(object):
    def __init__(self, camera_matrix, camera_distortion, marker_coords, reference_id):
        self.marker_coords = marker_coords
        self.camera_distortion = camera_distortion
        self.camera_matrix = camera_matrix
        self.reference_id = reference_id
        self.last_tvec = None
        self.last_rvec = None

    def get_frame_locations(self, markers, rvec=None, tvec=None):
        matrices = {}
        markers = {m['id']: m for m in markers}
        if self.reference_id not in markers: return {}
        for marker in markers.values():
            world_positions = self.marker_coords[marker['id']]
            screen_positions = marker['verts']
            retval, rvec, tvec = cv2.solvePnP(
                    world_positions, screen_positions.reshape(-1,1,2),
                    self.camera_matrix, self.camera_distortion)
            rotmat, jacobian = cv2.Rodrigues(rvec)
            mat = np.eye(4, 4)
            mat[:3, :3] = rotmat
            mat[:3, 3] = np.array(tvec).reshape(-1)
            matrices[marker['id']] = mat

        ref_matrix = matrices.pop(self.reference_id)
        positions = {}
        positions[self.reference_id] = self.marker_coords[self.reference_id]

        def multidot(M, vs):
            return np.array([np.dot(M, v) for v in vs])
        
        for marker_id, matrix in matrices.iteritems():
            homo = np.array([list(c) + [1] for c in self.marker_coords[marker_id]])
            camera = multidot(np.linalg.inv(matrix), homo)
            #camera /= camera[:,-1].reshape(-1, 1)
            ps = multidot(ref_matrix, camera)
            #ps /= ps[:,-1].reshape(-1, 1)
            #reproj_matrix = np.dot(np.linalg.inv(matrix), ref_matrix)
            #ps = np.inner(reproj_matrix, homo).T
            #ps = multidot(reproj_matrix, homo)
            #cam_coords = np.inner(homo, np.linalg.inv(matrix))
            #ps = np.inner(ref_matrix, cam_coords)
            positions[marker_id] = ps[:,:3]
        
        return positions

import scipy.optimize
def brutalizer(camera_matrix, camera_distortion, cam_poses, marker_poses, marker_geometry, marker_data):
    n_frames = len(marker_data)
    n_markers = len(marker_poses)
    params = np.hstack((np.ravel(marker_poses[1:]), np.ravel(cam_poses)))
    def unpack(x):
        transforms = x.reshape(-1, 2, 3)
        points = transforms[:(n_markers - 1)]
        cameras = transforms[(n_markers - 1):]
        return points, cameras
    #points, cameras = unpack(params)
    def project(x):
        point_trans, cameras = unpack(x)
        points = list(marker_geometry[0])
        for i, (rot, trans) in enumerate(point_trans):
            R = cv2.Rodrigues(rot)[0]
            points.extend(np.inner(R, marker_geometry[i + 1]).T + trans)
        points = np.array(points)
        error = 0.0
        npoints = 0
        diffs = []
        for (rot, trans), markers in zip(cameras, marker_data):
            est = cv2.projectPoints(points, rot, trans, camera_matrix, camera_distortion)[0]
            for m in markers:
                diffs.extend(est[m['id']] - m['verts'])
        errors = np.array(diffs).ravel()
        return errors
    res = scipy.optimize.least_squares(project, params, x_scale='jac', verbose=2, ftol=1e-16)
    point_trans, cameras = unpack(res.x)
    points = [marker_geometry[0]]
    import matplotlib.pyplot as plt
    for i, (rot, trans) in enumerate(point_trans):
        R = cv2.Rodrigues(rot)[0]
        est = np.inner(R, marker_geometry[i + 1]).T + trans
        #plt.plot(est[:,0], est[:,1])
        points.append(est)
    points = np.array(points)
    #plt.plot(points[:,0], points[:,1], '.')
    #plt.show()
    return points, cameras
            





def brutaltest():
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import pickle
    from simulate import markersim
    camera = pickle.load(open("rect_camera.pickle"))
    marker_geometry = np.array([
        [1,-1,0],
        [1,1,0],
        [-1,1,0],
        [-1,-1,0],
        ]).astype(np.float)*0.05
    mdict = defaultdict(lambda: marker_geometry)
    marker_data = markersim(camera['camera_matrix'], camera['dist_coefs'], mdict)
    marker_data = [marker_data.next() for i in range(10)]
    marker_data, tposes, tpositions = zip(*marker_data)
    marker_poses = []
    for tpos in tpositions[0]:
        marker_poses.append([[0.0, 0.0, 0.0], -np.mean(tpos, axis=0)+np.random.randn(3)])
    #marker_poses = np.zeros((len(tpositions[0]), 2, 3))
    marker_poses = np.array(marker_poses)
    p, c = brutalizer(camera['camera_matrix'], camera['dist_coefs'], tposes, marker_poses, mdict, marker_data)
    print np.round(np.mean(p, axis=1), 3)
    print marker_poses

def estimate_locations(marker_data, marker_geometry, camera_matrix, camera_distortion, reference_id=0):
    #marker_geometry = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]).astype(np.float)*0.05
    #marker_geometry = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]]).astype(np.float)*0.05
    
    #marker_geometry = np.array([[1,-1,0],[1,1,0],[-1,1,0],[-1,-1,0]]).astype(np.float)*0.5
    #marker_geometry -= marker_geometry[0]
    
    #marker_positions = defaultdict(lambda: marker_geometry)
    estimator = MarkerGeometryEstimator(camera_matrix, camera_distortion, marker_geometry, int(reference_id))
    
    estimates = defaultdict(list)
    for markers in marker_data:
        positions = estimator.get_frame_locations(markers)
        if len(positions) < 2: continue
        for key, pos in positions.iteritems():
            estimates[key].append(pos)
    coords = {}
    import matplotlib.pyplot as plt
    for key, estimate in estimates.iteritems():
        estimate = np.array(estimate)
        centests = np.mean(estimate, axis=1)
        if key <= 5:
            plt.plot(centests[:,0], centests[:,1], '.')
        center_median = np.median(centests, axis=0)
        errors = np.sqrt(np.sum((center_median - centests)**2, axis=1))
        mederror = np.median(errors)
        est = np.median(estimate, axis=0)
        #repro, jac = cv2.projectPoints(est, np.array([[0, 0, 0.0]]), np.array([[0, -0.5, 2.0]]), camera_matrix, camera_distortion)
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.plot(repro[:,0,0], repro[:,0,1])
        #plt.plot(repro[0,0,0], repro[0,0,1], 'o')
        coords[key] = {
                'positions': est,
                'n_observations': len(estimate),
                'median_error': mederror,
                }
    plt.show()
    return coords


def main(camera_spec, marker_file, output_file, reference_id=0):
    import pickle
    camera = pickle.load(open(camera_spec))
    marker_data = np.load(marker_file)
    
    coords = estimate_locations(marker_data,
            camera['camera_matrix'], camera['dist_coefs'],
            reference_id)
    pickle.dump(coords, open(output_file, 'w'))

if __name__ == '__main__':
    import argh
    #argh.dispatch_command(main)
    brutaltest()
