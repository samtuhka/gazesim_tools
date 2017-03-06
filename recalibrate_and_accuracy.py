import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
import cPickle as pickle
import mapping
import operator

from calibrate_3D import calibrate_3D, Binocular_Vector_Gaze_Mapper

conf = 0.6
 
def update(params, events, eye):
        map_fn = mapping.make_map_function(*params)
        gaze_pts = []
        for p in events:
            if p['confidence'] > conf and p['id']==eye:
                gaze_point = map_fn(p['norm_pos'])
                gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})
        return gaze_pts

def update_glint(params, events, eye):
        map_fn = mapping.make_map_function(*params)
        gaze_pts = []
        for p in events:
            if p['confidence'] > conf and p['id']==eye:
                glint = p['glints']
                if glint[0][2] > 0 and glint[1][2] > 0:
                     x = p["norm_pos"][0] - ((glint[0][3] + glint[1][3]) / 2.)
                     y = p["norm_pos"][1] - ((glint[0][4] + glint[1][4]) / 2.)
                     gaze_point = map_fn((x,y))
                     gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})
                elif glint[0][2] > 0:
                      x =  p["norm_pos"][0] - glint[0][3]
                      y =  p["norm_pos"][1] - glint[0][4]
                      gaze_point = map_fn((x,y))
                      gaze_pts.append({'norm_pos':gaze_point,'confidence':p['confidence'],'timestamp':p['timestamp'], 'id': p['id'], 'base':[p]})
        return gaze_pts


def update_binocular(params0, params1, events, params_bin, multivariate = False):
        map0 = mapping.make_map_function(*params0)
        map1 = mapping.make_map_function(*params1)

        map_bin = mapping.make_map_function(*params_bin)

        pupil_pts_0 = []
        pupil_pts_1 = []
        for p in events:
            if p['confidence'] > conf:
                if p['id'] == 0:
                    pupil_pts_0.append(p)
                else:
                    pupil_pts_1.append(p)

        gaze_pts = []
        p0 = pupil_pts_0.pop(0)
        p1 = pupil_pts_1.pop(0)
        while True:
                if len(pupil_pts_0) > 0 and len(pupil_pts_1) > 0:
                        gaze_point_eye0 = map0(p0['norm_pos'])
                        gaze_point_eye1 = map1(p1['norm_pos'])
                        gaze_point = (gaze_point_eye0[0] + gaze_point_eye1[0])/2. , (gaze_point_eye0[1] + gaze_point_eye1[1])/2.0
                if multivariate:
                        gaze_point = map_bin(p0['norm_pos'], p1['norm_pos'])
                confidence = (p0['confidence'] + p1['confidence'])/2.
                ts = (p0['timestamp'] + p1['timestamp'])/2.
                if abs(p0['timestamp'] - p1['timestamp']) < 1/40.:
                        gaze_pts.append({'norm_pos':gaze_point,'confidence':confidence,'timestamp':ts,'base':[p0, p1]})
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

def load_object(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path,'rb') as fh:
        return pickle.load(fh)

def save_object(object,file_path):
	file_path = os.path.expanduser(file_path)
	with open(file_path,'wb') as fh:
		pickle.dump(object,fh,-1)


def recalibrate(path):
        timestamps = np.load(path + "world_timestamps_unix.npy")
        timestamps2 = np.load(path + "world_timestamps.npy")
        start = timestamps[0]
        end=timestamps[-1]


        camera = load_object("camera_calibration")
        cam_dist = camera['dist_coefs']
        cam_dist = np.array([cam_dist[0][0], cam_dist[0][1], 0, 0, 0])
        cam_m = camera['camera_matrix']
        h, w = 720, 1280
        newcamera, roi = cv2.getOptimalNewCameraMatrix(cam_m, cam_dist, (w,h), 0)

        def undistort_calibration(cal_pt_cloud):
                for i in range(len(cal_pt_cloud)):
                        point = np.float32((cal_pt_cloud[i,2]*1280.0, 720 - cal_pt_cloud[i,3]*720)).reshape(-1,1,2)
                        new_point = tuple(cv2.undistortPoints(point,cam_m,cam_dist, P=newcamera).reshape(2))
                        cal_pt_cloud[i,2] = new_point[0]/1280.0
                        cal_pt_cloud[i,3] = (720.0 - new_point[1])/720.0
                return cal_pt_cloud


        valid_calibrations = []
        mappings = []
        rootdir = '/home/samtuhka/pupil/recordings/calibData/'
        for dirs in os.walk(rootdir):
            if not dirs[0]==rootdir:
                ts = float(os.path.basename(dirs[0]))
                if ts >= start and ts <= end:
                    valid_calibrations.append(dirs[0])
        if not valid_calibrations:
                valid_calibrations.append(path)



        pupil_3D_eye0 = load_object(path  + "recalculated_pupil_0")
        pupil_3D_eye1 = load_object(path  + "recalculated_pupil_1")
        pupil_list_3D = pupil_3D_eye0['pupil_positions'] + pupil_3D_eye1['pupil_positions']
        pupil_list_3D = sorted(pupil_list_3D, key=lambda k: k['timestamp'])

        data = load_object(path + 'pupil_data')
        pupil_list = data['pupil_list']
        pupil0 = [p for p in pupil_list if p['id']==0]
        pupil1 = [p for p in pupil_list if p['id']==1]
        
        for calib in valid_calibrations:
                try:
                        ref_list = np.load(calib + "/cal_ref_list.npy")
                except:
                        data = load_object(calib + "/user_calibration_data")
                        ref_list = data['ref_list']

                ts = ref_list[-1]['timestamp']

                params_3D = calibrate_3D(ref_list, pupil_list_3D)

                matched_pupil0_data = mapping.closest_matches_monocular(ref_list,pupil0)
                matched_pupil1_data = mapping.closest_matches_monocular(ref_list,pupil1)
                matched_binocular_data = mapping.closest_matches_binocular(ref_list,pupil_list)


                cal_pt_cloud0 = mapping.preprocess_2d_data_monocular(matched_pupil0_data)
                cal_pt_cloud1 = mapping.preprocess_2d_data_monocular(matched_pupil1_data)
                cal_pt_cloud_binocular = mapping.preprocess_2d_data_binocular(matched_binocular_data)

                map_fn,inliers,params = mapping.calibrate_2d_polynomial(cal_pt_cloud_binocular,(1280, 720),binocular=True)
                map_fn,inliers,params_eye0 = mapping.calibrate_2d_polynomial(cal_pt_cloud0,(1280, 720),binocular=False)
                map_fn,inliers,params_eye1 = mapping.calibrate_2d_polynomial(cal_pt_cloud1,(1280, 720),binocular=False)

                cal_pt_cloud0_glint = mapping.preprocess_2d_data_monocular_glint(matched_pupil0_data)
                cal_pt_cloud1_glint = mapping.preprocess_2d_data_monocular_glint(matched_pupil1_data)
                map_fn,inliers,params_eye0_glint = mapping.calibrate_2d_polynomial(cal_pt_cloud0_glint,(1280, 720),binocular=False)
                map_fn,inliers,params_eye1_glint = mapping.calibrate_2d_polynomial(cal_pt_cloud1_glint,(1280, 720),binocular=False)

                
                mappings.append([ts, params_eye0, params_eye1, params, params_eye0_glint, params_eye1_glint, params_3D])

                """
                if cal_pt_cloud0 and cal_pt_cloud1:
                        if len(cal_pt_cloud0) > max0:
                                max0 = len(cal_pt_cloud0)
                                map_fn,inliers,params_eye0 = mapping.calibrate_2d_polynomial(cal_pt_cloud0,(1280, 720),binocular=False)
                        if len(cal_pt_cloud1) > max1:
                                max1 = len(cal_pt_cloud1)
                                map_fn,inliers,params_eye1 = mapping.calibrate_2d_polynomial(cal_pt_cloud1,(1280, 720),binocular=False)
                                mappings = [[ts, params_eye0, params_eye1]]
                """
        
        mappings = sorted(mappings, key=operator.itemgetter(0))

        try:
                pupil_data = load_object(path + 'pupil_data_original')
        except:
                pupil_data = load_object(path + 'pupil_data')
                print "save original"
                save_object(pupil_data, path + "pupil_data_original")
        #glints = np.load(path + "glint_positions.npy")

        #eye0_glints = glints[glints[:,1]!=0]
        #eye1_glints = glints[glints[:,1]!=0]
        #eye0_glints = eye0_glints[eye0_glints[:,5]==0]
        #eye1_glints = eye1_glints[eye1_glints[:,5]==1]

        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update_binocular(mappings[i][1], mappings[i][2], pupil_positions,  mappings[i][3], True)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update_binocular(mappings[i][1], mappings[i][2], pupil_positions,  mappings[i][3], True)

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_binocular_multivariate")


        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update_binocular(mappings[i][1], mappings[i][2], pupil_positions,  mappings[i][3])
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update_binocular(mappings[i][1], mappings[i][2], pupil_positions,  mappings[i][3])

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_binocular")


        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update(mappings[i][1],pupil_positions, 0)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update(mappings[i][1], pupil_positions, 0)

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_0")

        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update(mappings[i][2], pupil_positions, 1)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update(mappings[i][2], pupil_positions, 1)

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_1")


        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update_glint(mappings[i][4],pupil_positions, 0)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update_glint(mappings[i][4], pupil_positions, 0)

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_0_glint")

        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_data['pupil_positions']:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        new_data += update_glint(mappings[i][5], pupil_positions, 1)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        new_data += update_glint(mappings[i][5], pupil_positions, 1)

        data = pupil_data['pupil_positions']
        pupil_data['gaze_positions'] = new_data
        save_object(pupil_data, path + "pupil_data_1_glint")



        pupil_positions = []
        new_data = []
        i = 0
        for p in pupil_list_3D:
                if i + 1 < len(mappings) and p['timestamp'] >= mappings[i+1][0]:
                        args = mappings[i][6]
                        mapper = Binocular_Vector_Gaze_Mapper(conf, args['eye_camera_to_world_matrix0'], args['eye_camera_to_world_matrix1'], args['camera_intrinsics'],
                                              args['cal_points_3d'], args['cal_ref_points_3d'], args['cal_gaze_points0_3d'], args['cal_gaze_points1_3d'])
                        new_data +=  mapper.update(pupil_positions)
                        i += 1
                        pupil_positions = []
                if p['timestamp'] >= mappings[i][0]:
                        pupil_positions.append(p)
        args = mappings[i][6]
        mapper = Binocular_Vector_Gaze_Mapper(conf, args['eye_camera_to_world_matrix0'], args['eye_camera_to_world_matrix1'], args['camera_intrinsics'],
                                              args['cal_points_3d'], args['cal_ref_points_3d'], args['cal_gaze_points0_3d'], args['cal_gaze_points1_3d'])
        new_data += mapper.update(pupil_positions)

        pupil_data['pupil_positions'] = pupil_list_3D
        pupil_data['gaze_positions'] = new_data

        save_object(pupil_data, path + "pupil_data_3D")

        




#recalibrate("/home/samtuhka/pupil/recordings/2016_06_21/009/")

