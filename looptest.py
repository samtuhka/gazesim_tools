import numpy as np
import cv2

geometry = np.array([[1,-1,0],[1,1,0],[-1,1,0],[-1,-1,0]]).astype(np.float)*0.05
geometry += np.array([1, 1, 2])

import pickle
camera = pickle.load(open('rect_camera.pickle'))
cm, cd = camera['camera_matrix'], camera['dist_coefs']

tvec = np.array([0, 0, 2.0]).astype(np.float).reshape(1, -1)
rvec = np.array([np.pi, 0, 0.0]).astype(np.float).reshape(1, -1)

screen, j = cv2.projectPoints(geometry, rvec, tvec, cm, cd)
ret, ervec, etvec = cv2.solvePnP(geometry, screen, cm, cd)
print np.round(ervec.T, 3)
print np.round(etvec.T, 3)
