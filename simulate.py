import numpy as np
import cv2
from estimateMarkerLocations import estimate_locations
from camera_poses import estimate_poses
from collections import defaultdict
import matplotlib.pyplot as plt

from markerloc import transform_points, compose_rotations

def markersim(camera_matrix, camera_distortion, marker_geometry, outlier_rate=0.1):
    np.random.seed(1)
    #marker_geometry = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]]).astype(np.float)*0.05
   
    #marker_geometry = marker_geometry[::-1]
    marker_origins = np.array([
        [0, 0.5, 0.0],
        [1, 0.5, 0.0],
        [-1, 0.5, 0.0],
        [1, -0.5, 0.0],
        [-1, -0.5, 0.0],
        ]).astype(np.float)
    
    #marker_origins = np.array([
    #    [0, 0.5, 0.0],
    #    [1, 0.5, 0.0],
    #    ]).astype(np.float)def transform_points(world, rvec, tvec):
    
    original_origin = marker_origins[0].copy()
    marker_origins -= original_origin
    marker_points = [marker_geometry[i] + o.reshape(1, -1) for i, o in enumerate(marker_origins)]
    #marker_points[1] = transform_points(marker_geometry[1], np.array([0, 0, 2*np.pi]), np.zeros(3)) + o[1]
    #marker_points[1] = transform_points(marker_geometry[1], np.array([0, 0, np.pi]), np.zeros(3)) + marker_origins[1]
    #marker_points[2] = transform_points(marker_geometry[2], np.array([0, 0, np.pi/2.0]), np.zeros(3)) + marker_origins[2]
    #camera_distortion = None
    t = 0
    while True:
        markers = []
        t += 0.1

        default_rvec = np.array([[0, 0, 0.0]])
        tvec = np.array([np.sin(t)*1, -0.5, 3.0]).astype(np.float).reshape(1, -1)
        tvec = np.array([[0, 0, 3.0]]) - original_origin# .astype(np.float)
        rvec = compose_rotations(
            default_rvec,
            np.array([np.sin(t*3)*0.5, np.sin(t*2)*0.5, 0])
            )
        
        #rvec += np.random.randn(*rvec.shape)*0.5
        #tvec += np.random.randn(*tvec.shape)*0.5
        #rvec = np.array([[  1.61344946e-03], [ -3.15707025e+00], [1.30836762e-01]])
        #tvec = np.array([[-0.11187565], [-0.48517735], [ 2.44699507]])
        for marker_id, o in enumerate(marker_points):
            if np.random.random() < outlier_rate:
                o = o + np.random.randn(1, 3)
            sp = cv2.projectPoints(o, rvec, tvec, camera_matrix, camera_distortion)[0]
            sp += np.random.randn(*sp.shape)*1.0
                
            #plt.plot(sp[:,0,0], sp[:,0,1])
            if np.any(sp[:,0, 0] < 0) or np.any(sp[:,0,0] > 1280): continue
            if np.any(sp[:,0,1] < 0) or np.any(sp[:,0,1] > 720): continue

            marker = {
                    'id': marker_id,
                    'verts': sp
                    }
            
            markers.append(marker)
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()
        frame = {
            'markers': markers,
            'ts': t
            }
        yield frame, (rvec.reshape(-1), tvec.reshape(-1)), marker_points

def main(camera_spec):
    import pickle
    camera = pickle.load(open(camera_spec))
    cm, cd = camera['camera_matrix'], camera['dist_coefs']
    
    marker_geometry = np.array([
        [1,-1,0],
        [1,1,0],
        [-1,1,0],
        [-1,-1,0],
        ]).astype(np.float)*0.05
    marker_geometry = marker_geometry[::-1]
    #marker_geometry2 = np.array([
    #    [1,-1,0],
    #    [1,1,0],
    #    [-1,1,0],
    #    [-1,-1,0],
    #    ]).astype(np.float)*0.025
    #marker_geometry -= marker_geometry[0]
    mdict = defaultdict(lambda: marker_geometry)

    sim = markersim(cm, cd, mdict)
    marker_data, tposes, tpositions = zip(*[sim.next() for i in range(10)])
    #marker_data = np.load('markers_long.npy')

    coords = estimate_locations(marker_data, mdict, cm, cd)
    from pprint import pprint
    ocoords = coords
    #ocoords = {k: l for k,l in coords.iteritems()
    #    if l['n_observations'] > 100}
    #ocoords = {i: {'positions': p, 'n_observations': np.inf, 'median_error': 0} for i, p in enumerate(tpositions[0])}
    import matplotlib.pyplot as plt
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'black']
    getcolor = lambda i: colors[i%len(colors)]
    for k, cs in ocoords.iteritems():
            ps = cs['positions']
            plt.plot(ps[:,0], ps[:,1], color=colors[k])
    #        tps = tpositions[0][k] - tpositions[0][0][0]
    #        plt.plot(tps[:,0], tps[:,1], '-', color=colors[k])
    plt.show()
    for i in range(len(ocoords)):
        coords = {k: v for k, v in ocoords.iteritems() if k == i}
        poses = estimate_poses(marker_data, coords, cm, cd)
        idx, rvecs, tvecs = map(np.array, zip(*poses))
        plt.plot(idx, tvecs[:,0])
    #trvecs, ttvecs = map(np.array, zip(*tposes))
    #plt.plot(idx, ttvecs[:,2])
    plt.show()

if __name__ == '__main__':
    import argh; argh.dispatch_command(main)
