import numpy as np
import mapping
import math_helper
from mapping import find_rigid_transform
from file_methods import load_object,save_object
from methods import project_distort_pts , normalize, spherical_to_cart
from optimization_calibration import  bundle_adjust_calibration
import cv2

import logging
logger = logging.getLogger(__name__)


def calibrate_3D(ref_list,pupil_list):

    camera_intrinsics = load_object("camera_calibration_test")

    pupil0 = [p for p in pupil_list if p['id']==0]
    pupil1 = [p for p in pupil_list if p['id']==1]


    matched_binocular_data = mapping.closest_matches_binocular(ref_list,pupil_list)
    print matched_binocular_data
    
    hardcoded_translation0  = np.array([20,15,-20])
    hardcoded_translation1  = np.array([-40,15,-20])
    
    method = 'binocular 3d model'

    smallest_residual = 1000
    scales = list(np.linspace(0.7,1.4,20))
    K = camera_intrinsics["camera_matrix"]

    for s in scales:
        scale = np.ones(K.shape)
        scale[0,0] *= s
        scale[1,1] *= s
        camera_intrinsics["camera_matrix"] = K*scale

        ref_dir, gaze0_dir, gaze1_dir = mapping.preprocess_3d_data(matched_binocular_data,
                                        camera_intrinsics = camera_intrinsics )

        if len(ref_dir) < 1 or len(gaze0_dir) < 1 or len(gaze1_dir) < 1:
            print(not_enough_data_error_msg)
            return

        sphere_pos0 = pupil0[-1]['sphere']['center']
        sphere_pos1 = pupil1[-1]['sphere']['center']

        initial_R0,initial_t0 = find_rigid_transform(np.array(gaze0_dir)*500,np.array(ref_dir)*500)
        initial_rotation0 = math_helper.quaternion_from_rotation_matrix(initial_R0)
        initial_translation0 = np.array(initial_t0).reshape(3)

        initial_R1,initial_t1 = find_rigid_transform(np.array(gaze1_dir)*500,np.array(ref_dir)*500)
        initial_rotation1 = math_helper.quaternion_from_rotation_matrix(initial_R1)
        initial_translation1 = np.array(initial_t1).reshape(3)

        eye0 = { "observations" : gaze0_dir , "translation" : hardcoded_translation0 , "rotation" : initial_rotation0,'fix':['translation']  }
        eye1 = { "observations" : gaze1_dir , "translation" : hardcoded_translation1 , "rotation" : initial_rotation1,'fix':['translation']  }
        world = { "observations" : ref_dir , "translation" : (0,0,0) , "rotation" : (1,0,0,0),'fix':['translation','rotation'],'fix':['translation','rotation']  }
        initial_observers = [eye0,eye1,world]
        initial_points = np.array(ref_dir)*500


        success,residual, observers, points  = bundle_adjust_calibration(initial_observers , initial_points, fix_points=False )

        if residual <= smallest_residual:
            smallest_residual = residual
            scales[-1] = s

    if not success:
        logger.error("Calibration solver faild to converge.")
        return


    eye0,eye1,world = observers


    t_world0 = np.array(eye0['translation'])
    R_world0 = math_helper.quaternion_rotation_matrix(np.array(eye0['rotation']))
    t_world1 = np.array(eye1['translation'])
    R_world1 = math_helper.quaternion_rotation_matrix(np.array(eye1['rotation']))

    def toWorld0(p):
        return np.dot(R_world0, p)+t_world0

    def toWorld1(p):
        return np.dot(R_world1, p)+t_world1

    points_a = [] #world coords
    points_b = [] #eye0 coords
    points_c = [] #eye1 coords
    for a,b,c,point in zip(world['observations'] , eye0['observations'],eye1['observations'],points):
        line_a = np.array([0,0,0]) , np.array(a) #observation as line
        line_b = toWorld0(np.array([0,0,0])) , toWorld0(b)  #eye0 observation line in world coords
        line_c = toWorld1(np.array([0,0,0])) , toWorld1(c)  #eye1 observation line in world coords
        close_point_a,_ =  math_helper.nearest_linepoint_to_point( point , line_a )
        close_point_b,_ =  math_helper.nearest_linepoint_to_point( point , line_b )
        close_point_c,_ =  math_helper.nearest_linepoint_to_point( point , line_c )
        points_a.append(close_point_a)
        points_b.append(close_point_b)
        points_c.append(close_point_c)


    sphere_translation = np.array( sphere_pos0 )
    sphere_translation_world = np.dot( R_world0 , sphere_translation)
    camera_translation = t_world0 - sphere_translation_world
    eye_camera_to_world_matrix0  = np.eye(4)
    eye_camera_to_world_matrix0[:3,:3] = R_world0
    eye_camera_to_world_matrix0[:3,3:4] = np.reshape(camera_translation, (3,1) )

    sphere_translation = np.array( sphere_pos1 )
    sphere_translation_world = np.dot( R_world1 , sphere_translation)
    camera_translation = t_world1 - sphere_translation_world
    eye_camera_to_world_matrix1  = np.eye(4)
    eye_camera_to_world_matrix1[:3,:3] = R_world1
    eye_camera_to_world_matrix1[:3,3:4] = np.reshape(camera_translation, (3,1) )


    args={
            'eye_camera_to_world_matrix0':eye_camera_to_world_matrix0,
            'eye_camera_to_world_matrix1':eye_camera_to_world_matrix1 ,
            'camera_intrinsics': camera_intrinsics ,
            'cal_points_3d': points,
            'cal_ref_points_3d': points_a,
            'cal_gaze_points0_3d': points_b,
            'cal_gaze_points1_3d': points_c}
    return args



class Binocular_Vector_Gaze_Mapper():
    def __init__(eye_camera_to_world_matrix0, eye_camera_to_world_matrix1 , camera_intrinsics , cal_points_3d = [],cal_ref_points_3d = [], cal_gaze_points0_3d = [], cal_gaze_points1_3d = [], conf ):

        self.eye_camera_to_world_matrix0  =  eye_camera_to_world_matrix0
        self.rotation_matrix0  =  eye_camera_to_world_matrix0[:3,:3]
        self.rotation_vector0 = cv2.Rodrigues( self.eye_camera_to_world_matrix0[:3,:3]  )[0]
        self.translation_vector0  = self.eye_camera_to_world_matrix0[:3,3]

        self.eye_camera_to_world_matrix1  =  eye_camera_to_world_matrix1
        self.rotation_matrix1  =  eye_camera_to_world_matrix1[:3,:3]
        self.rotation_vector1 = cv2.Rodrigues( self.eye_camera_to_world_matrix1[:3,:3]  )[0]
        self.translation_vector1  = self.eye_camera_to_world_matrix1[:3,3]

        self.cal_points_3d = cal_points_3d
        self.cal_ref_points_3d = cal_ref_points_3d

        self.cal_gaze_points0_3d = cal_gaze_points0_3d #save for debug window
        self.cal_gaze_points1_3d = cal_gaze_points1_3d #save for debug window

        self.camera_matrix = camera_intrinsics['camera_matrix']
        self.dist_coefs = camera_intrinsics['dist_coefs']
        self.camera_intrinsics = camera_intrinsics
        self.conf = conf

        self.gaze_pts_debug0 = []
        self.gaze_pts_debug1 = []
        self.intersection_points_debug = []
        self.sphere0 = {}
        self.sphere1 = {}
        self.last_gaze_distance = 0.0


    def update(self,events):

        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events['pupil_positions']:
            if p['confidence'] > self.conf:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        # try binocular mapping (needs at least 1 pupil position in each list)
        gaze_pts = []
        if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
            gaze_pts = self.map_binocular_intersect(pupil_pts_0, pupil_pts_1)
            events['gaze_positions'] = gaze_pts


    def eye0_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix0 , point)[:3]

    def eye1_to_World(self, p):
        point = np.ones(4)
        point[:3] = p[:3]
        return np.dot(self.eye_camera_to_world_matrix1 , point)[:3]

    def map_binocular_intersect(self, pupil_pts_0, pupil_pts_1):
        # maps gaze with binocular mapping
        # requires each list to contain at least one item!
        # returns 1 gaze point at minimum
        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:

            #find the nearest intersection point of the two gaze lines
            # a line is defined by two point
            s0_center = self.eye0_to_World( np.array( p0['sphere']['center'] ) )
            s1_center = self.eye1_to_World( np.array( p1['sphere']['center'] ) )

            s0_normal = np.dot( self.rotation_matrix0, np.array( p0['circle_3d']['normal'] ) )
            s1_normal = np.dot( self.rotation_matrix1, np.array( p1['circle_3d']['normal'] ) )

            gaze_line0 = [ s0_center, s0_center + s0_normal ]
            gaze_line1 = [ s1_center, s1_center + s1_normal ]

            nearest_intersection_point , intersection_distance = math_helper.nearest_intersection( gaze_line0, gaze_line1 )

            if nearest_intersection_point is not None :

                self.last_gaze_distance = np.sqrt( nearest_intersection_point.dot( nearest_intersection_point ) )

                image_point, _  =  cv2.projectPoints( np.array([nearest_intersection_point]) ,  np.array([0.0,0.0,0.0]) ,  np.array([0.0,0.0,0.0]) , self.camera_matrix , self.dist_coefs )
                image_point = image_point.reshape(-1,2)
                image_point = normalize( image_point[0], (1280, 720) , flip_y = True)

                confidence = (p0['confidence'] + p1['confidence'])/2.
                ts = (p0['timestamp'] + p1['timestamp'])/2.
                if abs(p0['timestamp'] - p1['timestamp']) < 1/40.:
                    gaze_pts.append({   'norm_pos':image_point,
                                        'eye_centers_3d':{0:s0_center.tolist(),1:s1_center.tolist()},
                                        'gaze_normals_3d':{0:s0_normal.tolist(),1:s1_normal.tolist()},
                                        'gaze_point_3d':nearest_intersection_point.tolist(),
                                        'confidence':confidence,
                                        'timestamp':ts,
                                        'base':[p0,p1]})



            # keep sample with higher timestamp and increase the one with lower timestamp
            if p0['timestamp'] <= p1['timestamp'] and pupil_pts_0:
                p0 = pupil_pts_0.pop(0)
                continue
            elif p1['timestamp'] <= p0['timestamp'] and pupil_pts_1:
                p1 = pupil_pts_1.pop(0)
                continue
            elif pupil_pts_0 and not pupil_pts_1:
                p0 = pupil_pts_0.pop(0)
            elif pupil_pts_1 and not pupil_pts_0:
                p1 = pupil_pts_1.pop(0)
            else:
                break

        return gaze_pts
