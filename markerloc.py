#!/usr/bin/env python2
from __future__ import division

from collections import defaultdict, OrderedDict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
import pickle
import cv2
import scipy.optimize
import scipy.sparse
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging
import marker_ba as mba
#logger.setLevel(logging.DEBUG)

from camera_poses import estimate_poses
def pixerrors(diffs):
    diffs = np.sqrt(np.sum(diffs**2, axis=1))
    return diffs

def transform_points(world, rvec, tvec):
    angle = np.linalg.norm(rvec)
    if angle > 1e-6:
        axis = (rvec/angle).reshape(3, 1)
        x, y, z = axis.reshape(-1)
        R = np.array((((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))))
        R = np.cos(angle)*np.identity(3) + np.sin(angle)*R + (1 - np.cos(angle))*np.dot(axis, axis.T)
        world = np.inner(R, world).T
    tvec = tvec.reshape(-1)
    return world + tvec

def compose_rotations(r1, r2):
    M1 = cv2.Rodrigues(r1)[0]
    M2 = cv2.Rodrigues(r2)[0]
    return cv2.Rodrigues(np.dot(M1, M2))[0]

def getProjectionMatrix(cm, rvec, tvec):
    T = np.empty((3,4))
    T[:,:3] = cv2.Rodrigues(rvec)[0]
    T[:,-1] = tvec.reshape(-1)
    return np.dot(cm, T)

def project_points(world, rvec, tvec, cm, cd):
    #angle = np.linalg.norm(rvec)
    #axis = (rvec/angle).reshape(3, 1)
    #x, y, z = axis.reshape(-1)
    #R = np.array((((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))))
    #R = np.cos(angle)*np.identity(3) + np.sin(angle)*R + (1 - np.cos(angle))*np.dot(axis, axis.T)
    #tvec = tvec.reshape(-1)
    #world = np.inner(R, world).T + tvec
    world = transform_points(world, rvec, tvec)
    proj = np.inner(cm, world).T
    proj /= proj[:,-1:]
    return proj[:,:2]

def project_frames(poses, world, cm, cd):
    ests = np.empty((len(poses), len(world), 2))
    #ests = []
    for i, (rvec, tvec) in enumerate(poses):
        est = cv2.projectPoints(world, rvec, tvec, cm, cd)[0].reshape(-1, 2)
        #est = project_points(world, rvec, tvec, cm, cd)
        #ests.append(est)
        ests[i] = est
    #print len(ests[i])
    return np.array(ests)

def reprojection_errors(poses, world, screen, cm, cd):
    ests = project_frames(poses, world, cm, cd)
    screen = screen.reshape(-1, len(world), 2)
    ests -= screen.reshape(-1, len(world), 2)
    """
    center = np.array([cm[0][-1], cm[1][-1]]).reshape(1, 1, 2)
    scale = np.array([cm[0][0], cm[1][1]]).reshape(1, 1, 2)
    offset = screen - center
    offset /= scale
    offset = np.exp(-(np.sum(offset**2, axis=2))*10).reshape(-1, 4, 1)
    """
    # Downweigh eccentric points
    #ests *= offset
    return ests

def estimate_marker_location(screens, poses, mpose, mworld, cm, cd):
    mposes = []
    mscreens = []
    for i, rvec, tvec in poses:
        if i not in screens: continue
        mposes.append((rvec, tvec))
        mscreens.append(screens[i])
    mscreens = np.array(mscreens)
    def errfunc(mpose):
        world = get_marker_points(mworld, mpose)
        errors = reprojection_errors(mposes, world, mscreens, cm, cd).ravel()
        #errors /= len(errors)
        return errors
        edges = []
        for i in range(len(world)):
            edge = world[i] - world[(i+1)%len(world)]
            edgelen = np.sqrt(np.sum(edge**2))
            edges.append(edgelen)
        evar = np.var(edges)*5
        ones = np.ones((world.shape[-1], 1))
        tmp = np.ones((4, 4))
        tmp[:,:-1] = world
        det = np.linalg.det(tmp.T)*5
        return errors
        #return np.hstack((errors, [evar, det]))
    
    #jac = jacobian(errfunc)
    """
    print jac(world.ravel())
    jac(world.ravel())
    print "Done"
    """
    
    result = scipy.optimize.least_squares(errfunc, mpose.ravel(), x_scale='jac', loss='huber')
    e = errfunc(result.x)
    deviations = e.reshape(-1, 2) #+ np.abs(e[-2]) + np.abs(e[-1])
    return result.x, deviations


def estimate_marker_locations(marker_data, poses, estimated_locations, mdict, cm, cd):
    screens = defaultdict(dict)
    for i, markers in enumerate(marker_data):
        for marker in markers:
            screens[marker['id']][i] = marker['verts']
    
    new_estimated_locations = {}
    errors = {}
    for marker_id, world in estimated_locations.iteritems():
        est, err = estimate_marker_location(screens[marker_id], poses, world, mdict[marker_id], cm, cd)
        new_estimated_locations[marker_id] = est
        errors[marker_id] = err
    return new_estimated_locations, errors

from mpl_toolkits.mplot3d import Axes3D
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def initial_marker_pose():
    p = np.zeros(1+3*2)
    return p

def get_marker_points(geom, pose):
    return transform_points(geom*(np.exp(pose[0]) + 0.1), *pose[1:].reshape(2, 3))

"""
def initial_marker_pose():
    return np.zeros(3*2)
def get_marker_points(geom, pose):
    return transform_points(geom, *pose.reshape(2, 3))
"""

"""
def initial_marker_pose():
    return np.zeros(3)
def get_marker_points(geom, pose):
    return geom + pose
"""

class _Slice(object):
    def __init__(self, slicer, v):
        self.slicer = slicer
        self.shape = v.shape
        i = len(self.slicer.d)
        self.slicer.d.extend(v.ravel())
        self.slc = slice(i, len(self.slicer.d))

    def get(self):
        return np.reshape(self.slicer.d[self.slc], self.shape)

    def set(self, value):
        shape = value.shape
        self.slicer.d[self.slc] = value.ravel()
        self.shape = shape

class Slicer(object):
    def __init__(self, i=0):
        self.d = []
    
    def __call__(self, v):
        return _Slice(self, v)

class FakeSlice(object):
    def __init__(self, v):
        self.v = v
        self.slc = slice(0, 0)

    def get(self):
        return self.v

    def set(self, v):
        self.v = v

def sliceidx(slc):
    step = slc.step
    if step is None: step = 1
    return range(slc.start, slc.stop, step)

def subslice(orig, start, n):
    return slice(orig.start, orig.start+n)

def robust_marker_pose_blocks(cm, cd, marker_data, marker_positions, initial_poses=None, blocksize=30, **kwargs):
    camposes = OrderedDict()
    queue = marker_data
    total = len(queue)
    i = 0
    while i < total:
        block = marker_data[i:i+blocksize]
        blockposes, result = robust_marker_pose_full(cm, cd, block, marker_positions, initial_poses,
                initial_i=i, **kwargs)
        logging.info("%i poses estimated"%(i + len(blockposes)))
        i += blocksize
        camposes.update(blockposes)

    return camposes

def robust_marker_pose(cm, cd, marker_data, marker_positions, jerk_std=(0.01, 0.01), **kwargs):
    poses = robust_marker_pose_full(cm, cd, marker_data, marker_positions, jerk_std=None, **kwargs)[0]
    return poses
    print "First pass done"
    if jerk_std is None: return poses
    kwargs['initial_poses'] = poses
    poses, result = robust_marker_pose_full(cm, cd, marker_data, marker_positions,
            jerk_std=jerk_std, **kwargs)
    print "Second pass done"
    return poses
    
def robust_marker_pose_full(cm, cd, marker_data, marker_positions, initial_poses=None,
        pixel_std=1.0, jerk_std=(0.01, 0.01), initial_i=0):
    xi = Slicer()
    ei = Slicer()
    if initial_poses is None:
        initial_poses = defaultdict(lambda: np.array([[0, 0, 0], [0, 0, -2]]).astype(np.float))
    frames = OrderedDict()
    for i, markers in enumerate(marker_data, initial_i):
        campose = xi(initial_poses[i])
        screens = []
        worlds = []
        for marker in markers:
            mid = marker['id']
            try:
                p = marker_positions[mid]
            except KeyError:
                continue
            screens.extend(marker['verts'].reshape(-1, 2))
            worlds.extend(p)
        if len(screens) == 0: continue
        screens = np.array(screens)
        worlds = np.array(worlds)
        errors = ei(np.zeros(screens.shape))
        if not jerk_std:
            transition = FakeSlice(np.zeros(1))
        else:
            transition = ei(np.zeros(1))
        frames[i] = (campose, worlds, screens, errors, transition)
    
    nonzero_jac = scipy.sparse.lil_matrix((len(ei.d), len(xi.d)))
    pframe = None
    ppframe = None
    for i, frame in enumerate(frames.itervalues()):
        (campose, world, screen, errors, transition) = frame
        nonzero_jac[errors.slc, campose.slc] = 1
        #nonzero_jac[transition.slc, campose.slc] = 1
        if not jerk_std: continue
        #if pframe:
        #    nonzero_jac[transition.slc, pframe[0].slc] = 1
        #if ppframe:
        #    nonzero_jac[transition.slc, ppframe[0].slc] = 1
        ppframe = pframe
        pframe = frame
        # The smoothing works surprisingly well without
        # having it in the jacobian
            
    
    xi.d = np.array(xi.d)
    ei.d = np.array(ei.d)
    call_n = [0]
    def reproError(d):
        xi.d = d.copy()
        pframe = None; ppframe = None; pppframe = None
        for i, frame in enumerate(frames.itervalues()):
            (campose, world, screen, errors, transition) = frame
            rvec, tvec = campose.get()
            pix_error = cv2.projectPoints(world, rvec, tvec, cm, cd)[0].reshape(-1, 2)
            pix_error -= screen
            pix_error /= len(pix_error)
            pix_error /= pixel_std
            errors.set(pix_error)
            if jerk_std is None: continue
            if pframe is not None and ppframe is not None and pppframe is not None:
                speed = np.linalg.norm(pframe[0].get()[0])
                pspeed = np.linalg.norm(ppframe[0].get()[0] - pframe[0].get()[0])
                ppspeed = np.linalg.norm(pppframe[0].get()[0] - ppframe[0].get()[0])
                accel = speed - pspeed
                paccel = ppspeed - pspeed
                jerk = np.linalg.norm(accel - paccel)
                transition.set(jerk/jerk_std[0])
            pppframe = ppframe
            ppframe = pframe
            pframe = frame
        call_n[0] += 1
        err = ei.d.copy()
        if call_n[0]%100 == 0:
            merr = np.mean(np.abs(err))
            Merr = np.mean(np.median(np.abs(err)))
            logging.debug("Pose: Error %.2f/%.2f at iteration %i"%(merr, Merr, call_n[0]))

        return err
    
    result = scipy.optimize.least_squares(reproError, xi.d,
            jac_sparsity=nonzero_jac,
            jac='3-point',
            x_scale='jac',
            ftol=1e-6,
            #loss='soft_l1',
            loss='huber',
            #verbose=2
            )
    xi.d = result.x
    camposes = OrderedDict((i, f[0].get()) for i, f in frames.iteritems())
    return camposes, result

def marker_point_ba(cm, cd, marker_data, init_camposes, init_marker_points, reference_id,
        fix_cameras=False, fix_points=False):
    xi = Slicer()
    ei = Slicer()
    frames = OrderedDict()
    marker_points = {}
    marker_points[reference_id] = FakeSlice(init_marker_points[reference_id])
    has_reference_marker = False

    for i, markers in enumerate(marker_data):

        markerpix = {}
        marker_errors = {}
        for marker in markers:
            mid = marker['id']
            if mid in markerpix:
                print "Duplicate marker", mid
                continue

            if mid == reference_id:
                has_reference_marker = True
            if mid not in marker_points:
                try:
                    p = init_marker_points[mid]
                except KeyError:
                    continue
                if not fix_points:
                    marker_points[mid] = xi(p)
                else:
                    marker_points[mid] = FakeSlice(p)
            markerpix[mid] = marker['verts'].reshape(-1, 2)
            marker_errors[mid] = ei(np.zeros(markerpix[mid].shape))
        if len(markerpix) == 0:
            continue
        try:
            ipose = init_camposes[i]
        except KeyError:
            continue
        if not fix_cameras:
            campose = xi(ipose)
        else:
            campose = FakeSlice(ipose)
        frames[i] = (campose, markerpix, marker_errors)
    
    if not has_reference_marker:
        raise ValueError("No reference marker (id: %i) found in dataset!"%reference_id)
    xi.d = np.array(xi.d)
    ei.d = np.array(ei.d)
    if len(xi.d) == 0 or len(ei.d) == 0:
        return init_camposes, init_marker_points, None
    nonzero_jac = scipy.sparse.lil_matrix((len(ei.d), len(xi.d)))
    for (cam_pose, markerpix, marker_errors) in frames.values():
        for k, me in marker_errors.iteritems():
            nonzero_jac[me.slc, cam_pose.slc] = 1
            nonzero_jac[me.slc, marker_points[k].slc] = 1
            #for mi in range(len(marker_points[k].get())):
            #    nonzero_jac[
            #            subslice(me.slc, mi*2, 2),
            #            subslice(marker_points[k].slc, mi*3, 3)] = 1
    logger.info("Optimizing %i frames with %i markers"%(len(frames), len(marker_points)))
    print len(xi.d), len(ei.d)
    call_n = [0]
    
    def loss(e2):
        loss = np.zeros((3, len(e2)))
        print e2
        loss[0,:] = e2
        return loss

    def reproError(d):
        xi.d = d
        for (cam_pose, markerpix, marker_errors) in frames.values():
            rvec, tvec = cam_pose.get()
            for mid, s in markerpix.iteritems():
                world = marker_points[mid].get()
                
                est = cv2.projectPoints(world, rvec, tvec, cm, cd)[0].reshape(-1, 2)
                depths = transform_points(world, rvec, tvec)[:,2]
                depths[depths > 0] = 0
                err = est - s
                # As a hack, add a huge penalty for points behind the camera
                err[:,0] -= np.sign(err[:,0])*depths*1e6
                err[:,1] -= np.sign(err[:,1])*depths*1e6
                marker_errors[mid].set(err)
        err = ei.d.copy()
        merr = np.mean(np.abs(err))
        Merr = np.percentile(np.abs(err), 75)
        if call_n[0]%100 == 0:
            logging.debug("Error %.10f/%.10f at iteration %i"%(merr, Merr, call_n[0]))
        call_n[0] += 1
        return err
    result = scipy.optimize.least_squares(reproError, xi.d,
            jac_sparsity=nonzero_jac,
            #jac='3-point',
            x_scale='jac',
            #loss='soft_l1', f_scale=10.0,
            loss='huber',f_scale=2,
            #loss=loss,
            ftol=1e-4,
            #xtol=1e-4,
            verbose=2
            )
    xi.d = result.x
    camposes = OrderedDict((i, f[0].get()) for i, f in frames.iteritems())
    marker_points = {k: v.get() for k, v in marker_points.iteritems()}
    return camposes, marker_points, result



def marker_ba(cm, cd, marker_data, init_camposes, mdict, init_markerposes, reference_id,
                fix_cameras=False, fix_markers=False):
    xi = Slicer()
    ei = Slicer()
    frames = OrderedDict()
    markerposes = {}
    markerposes[reference_id] = FakeSlice(init_markerposes[reference_id])
    has_reference_marker = False

    for i, markers in enumerate(marker_data):
        markerpix = {}
        marker_errors = {}
        fmposes = {}
        for marker in markers:
            mid = marker['id']
            if mid == reference_id:
                has_reference_marker = True
            if mid in markerpix:
                continue
            if mid not in markerposes:
                try:
                    p = init_markerposes[mid]
                except KeyError:
                    continue
                if not fix_markers:
                    markerposes[mid] = xi(p)
                else:
                    markerposes[mid] = FakeSlice(p)
            markerpix[mid] = marker['verts'].reshape(-1, 2)
            marker_errors[mid] = ei(np.zeros(markerpix[mid].shape))
        if len(markerpix) == 0:
            continue
        try:
            campose = init_camposes[i]
        except KeyError:
            continue
        if not fix_cameras:
            campose = xi(campose)
        else:
            campose = FakeSlice(campose)


        frames[i] = (campose, markerpix, marker_errors)

    if not has_reference_marker:
        raise ValueError("No reference marker (id: %i) found in dataset!"%reference_id)
    xi.d = np.array(xi.d)
    ei.d = np.array(ei.d)
    
    if len(xi.d) == 0 or len(ei.d) == 0:
        return init_camposes, init_markerposes, None

    nonzero_jac = scipy.sparse.lil_matrix((len(ei.d), len(xi.d)))
    for (cam_pose, markerpix, marker_errors) in frames.values():
        for k, me in marker_errors.iteritems():
            nonzero_jac[me.slc, cam_pose.slc] = 1
            nonzero_jac[me.slc, markerposes[k].slc] = 1
    logger.info("Optimizing %i frames with %i markers"%(len(frames), len(markerposes)))
    call_n = [0]
    def errdump(err, iteration):
        aerr = np.abs(err)
        logging.debug("Error mean/median %.3f/%.3f at iteration %s"%(np.mean(aerr), np.median(aerr), iteration))
    
    def reproError(d):
        xi.d = d.copy()
        for (cam_pose, markerpix, marker_errors) in frames.values():
            rvec, tvec = cam_pose.get()
            for mid, s in markerpix.iteritems():
                pose = markerposes[mid].get()
                world = get_marker_points(mdict[mid], pose)
                est = cv2.projectPoints(world, rvec, tvec, cm, cd)[0].reshape(-1, 2)
                depths = transform_points(world, rvec, tvec)[:,2]
                depths[depths > 0] = 0
                err = est - s
                # As a hack, add a huge penalty for points behind the camera
                err[:,0] -= np.sign(err[:,0])*depths*1e6
                err[:,1] -= np.sign(err[:,1])*depths*1e6
                marker_errors[mid].set(err)
        err = ei.d.copy()
        if call_n[0]%100 == 0:
            errdump(err, call_n[0])
        call_n[0] += 1
        return err
        
    result = scipy.optimize.least_squares(reproError, xi.d,
            jac_sparsity=nonzero_jac,
            jac='3-point',
            x_scale='jac',
            #loss='soft_l1',
            #loss='huber', f_scale=1.0,
            ftol=1e-4,
            #verbose=2
            )
    xi.d = result.x
    errdump(ei.d, "%i FINAL"%call_n[0])
    camposes = OrderedDict((i, f[0].get()) for i, f in frames.iteritems())
    markerposes = {k: v.get() for k, v in markerposes.iteritems()}
    return camposes, markerposes, result


class TriangulationError(Exception): pass
def multiviewTriangulation(cm, poses, points, max_frames=100):
    projections = (getProjectionMatrix(cm, *p) for p in itertools.islice(poses, max_frames))
    pp = zip(projections, points)
    ests = []
    for (P1, x1), (P2, x2) in itertools.combinations(pp, 2):
        est = cv2.triangulatePoints(P1, P2, x1.T, x2.T).T
        est = est[:,:-1]/est[:,-1:]
        ests.append(est)

    if len(ests) == 0:
        raise TriangulationError("Not enough poses")
    ests = np.array(ests)
    mest = np.median(ests, axis=0)
    #plt.plot(mest[:,0], mest[:,1])
    #plt.plot(ests[:,0,0], ests[:,0,1], '.')
    #plt.plot(ests[:,1,0], ests[:,1,1], '.')
    #plt.plot(ests[:,2,0], ests[:,2,1], '.')
    #plt.plot(ests[:,3,0], ests[:,3,1], '.')
    #plt.show()
    return mest

def multiviewMarkerTriangulation(cm, poses, marker_frames):
    poselist = []
    pointslist = []
    for i, pose in poses.iteritems():
        try:
            marker = marker_frames[i]
        except KeyError:
            continue    
        poselist.append(pose)
        pointslist.append(marker['verts'].reshape(-1, 2))
    return multiviewTriangulation(cm, poselist, pointslist)

def rough_marker_pose_estimate_full(cm, cd, marker_points, marker_frames):
    poses = {}
    for i, frame in marker_frames.iteritems():
        try:
            poses[i] = rough_marker_pose_estimate(cm, cd, marker_points, frame)
        except PoseEstimateError, e:
            continue
    return poses

class PoseEstimateError(Exception): pass
def rough_marker_pose_estimate(cm, cd, marker_points, frame):
    worlds = []
    screens = []
    for mid, world in marker_points.iteritems():
        if mid not in frame: continue
        screens.extend(frame[mid])
        worlds.extend(world)
    
    if len(worlds) < 4:
        raise PoseEstimateError("Not enough points for pose estimate")
    worlds = np.array(worlds)
    screens = np.array(screens)
    ret, rvec, tvec = cv2.solvePnP(worlds, screens, cm, cd)
    if not ret:
        raise PoseEstimateError("Couldn't estimate pose")
    return (rvec.T, tvec.T)


def plot_poses(poses, fmt='.'):
    acolors = ['red', 'green', 'blue']
    #camera_poses = robust_marker_pose(cm, cd, marker_data, pointify(marker_poses))
    for i, color in enumerate(acolors):
        plt.subplot(2,1,1)
        plt.plot(poses.keys(), [k[1].reshape(-1)[i] for k in poses.values()], fmt, color=acolors[i])
        plt.subplot(2,1,2)
        plt.plot(poses.keys(), [k[0].reshape(-1)[i] for k in poses.values()], fmt, color=acolors[i])

def plot_markers(marker_points):
    ax = plt.gcf().add_subplot(111, projection='3d')
    #acolors = ['red', 'green', 'blue']
    #camera_poses = robust_marker_pose(cm, cd, marker_data, pointify(marker_poses))
    for marker_id, res in marker_points.iteritems():
        ax.text(res[0,0], res[0,1], res[0,2], str(marker_id))
        ax.plot(res[:,0], res[:,1], res[:,2], color='red')
    axisEqual3D(ax)

def test(camera_spec, marker_file, out_file, reference_id=0, frequency_threshold=0.05,
        max_frames=300, error_threshold=5.0, visualize=False):
    camera = pickle.load(open(camera_spec))
    cm, cd = camera['camera_matrix'], camera['dist_coefs']
    marker_geometry = np.array([
        [-1,1,0],
        [1,1,0],
        [1,-1,0],
        [-1,-1,0],
        ]).astype(np.float)*0.05
    mdict = defaultdict(lambda: marker_geometry)
    def pointify(mposes):
        return {k: get_marker_points(mdict[k], pose) for k, pose in mposes.iteritems()}
    
    #marker_geometry = marker_geometry[::-1]
    #a = np.pi/2
    #m = marker_geometry
    #m[:,0], m[:,1] = (
    #        np.cos(a)*m[:,0] - np.sin(a)*m[:,1],
    #        np.sin(a)*m[:,0] + np.cos(a)*m[:,1],
    #        )
    
    from simulate import markersim
    omarker_data, tposes, tpositions = zip(*itertools.islice(markersim(cm, cd, mdict), 0, max_frames)); reference_id=0
    marker_data = omarker_data
    global plot_poses
    _plot_poses = plot_poses
    def plot_poses(poses):
        _plot_poses(OrderedDict(enumerate(tposes)), '-')
        _plot_poses(poses)
    #marker_data = omarker_data
    #marker_data = omarker_data[:100]
    
    #reference_id = 19
    #whitelist = [reference_id, 7]
    #reference_id = 7
    whitelist = [6, 19]
    #reference_id = 6
    #whitelist = [6]
    #whitelist = [6]
    #reference_id = whitelist[0]

    """
    omarker_data = np.load(marker_file)
    decim = len(omarker_data)//max_frames + 1
    #marker_data = omarker_data[::decim]
    marker_data = omarker_data[:max_frames]
    """
    marker_data_clean = OrderedDict()
    frame_times = OrderedDict()
    for i, frame in enumerate(marker_data):
        clean_frame = marker_data_clean[i] = {}
        frame_times[i] = frame['ts']
        for marker in frame['markers']:
            clean_frame[marker['id']] = marker['verts'].reshape(-1, 2).astype(np.float64)
    #marker_data = omarker_data[10:max_frames]
    
    #marker_data = omarker_data[:100]

    camera_poses = defaultdict(lambda: np.array([[0, 0, np.pi], [0, 0, 2]]).astype(np.float))
    marker_poses = defaultdict(initial_marker_pose)

    marker_frames = defaultdict(OrderedDict)
    for i, frame in enumerate(marker_data):
        for marker in frame['markers']:
            marker_frames[marker['id']][i] = marker
    
    #marker_poses = {k: initial_marker_pose() for k in whitelist}
    #camera_poses, marker_poses, _ = marker_ba(cm, cd, marker_data, camera_poses, mdict, marker_poses, reference_id)
    #marker_points = pointify(marker_poses)
    
    
    #camera_poses, marker_points, _ = marker_point_ba(cm, cd, marker_data, camera_poses, marker_points, reference_id)
    
    #marker_points = {k: mdict[k] for k in whitelist}
    
    marker_points = mdict.copy()
    #marker_points = {k: mdict[k] for k in whitelist}
    known_marker_points = {reference_id: marker_points[reference_id]}
    known_camera_poses = rough_marker_pose_estimate_full(cm, cd, known_marker_points, marker_data_clean)
    #plot_poses(camera_poses)
    #plt.show()
    cameras_known = 0
    while True:
        #plt.suptitle("Rough"); plot_poses(known_camera_poses); plt.show()
        known_camera_poses, _ = mba.marker_point_ba(cm, marker_data_clean, frame_times,
                known_camera_poses, known_marker_points, reference_id, fix_features=True)
        
        #plt.suptitle("Optimized"); plot_poses(known_camera_poses); plt.show()
        for mid, frames in marker_frames.iteritems():
            if mid in known_marker_points: continue
            try:
                known_marker_points[mid] = multiviewMarkerTriangulation(cm, known_camera_poses, frames)
            except TriangulationError, e:
                continue

        _, known_marker_points = mba.marker_point_ba(cm, marker_data_clean, frame_times,
                known_camera_poses, known_marker_points, reference_id, fix_cameras=True)
        
        for fid, frame in marker_data_clean.iteritems():
            if mid in known_camera_poses: continue
            try:
                known_camera_poses[mid] = rough_marker_pose_estimate(cm, cd, known_marker_points, frame)
                print "Got new estimate"
            except PoseEstimateError, e:
                continue

        known_camera_poses, known_marker_points = mba.marker_point_ba(
                cm,
                marker_data_clean, frame_times,
                known_camera_poses,
                known_marker_points,
                reference_id
                )

        if cameras_known >= len(known_camera_poses):
            break
        cameras_known = len(known_camera_poses)
    
    known_camera_poses, _ = mba.marker_point_ba(cm, marker_data_clean, frame_times,
        known_camera_poses, known_marker_points, reference_id, fix_features=True)
    
    camera_poses = known_camera_poses
    marker_points = known_marker_points
    
    """
    #marker_poses = {k: initial_marker_pose() for k in whitelist}
    known_marker_poses = {reference_id: marker_poses[reference_id]}
    camera_poses.update(rough_marker_pose_estimate(cm, cd, mdict[reference_id], marker_frames[reference_id]))
    while True:
        #known_camera_poses = (robust_marker_pose(cm, cd, marker_data, pointify(known_marker_poses), initial_poses=camera_poses))
        known_camera_poses, _, result = marker_ba(cm, cd, marker_data,
                camera_poses, mdict, known_marker_poses, reference_id, fix_markers=True)
        camera_poses.update(known_camera_poses)
        _, known_marker_poses, result = marker_ba(cm, cd, marker_data,
                known_camera_poses, mdict, marker_poses, reference_id, fix_cameras=True)
        marker_poses.update(known_marker_poses)
        known_camera_poses, known_marker_poses, result = marker_ba(cm, cd, marker_data,
                known_camera_poses, mdict, known_marker_poses, reference_id)
        camera_poses.update(known_camera_poses)
        marker_poses.update(known_marker_poses)
        if cameras_known == len(camera_poses):
            break
        cameras_known = len(camera_poses)
    camera_poses = known_camera_poses
    marker_poses = known_marker_poses
    marker_points = pointify(marker_poses)
    #plt.plot(camera_poses.keys(), [k[1][0] for k in camera_poses.values()], color='red')
    #plt.show()

    #camera_poses, marker_poses, result = marker_ba(cm, cd, marker_data[::10], camera_poses, mdict, marker_poses, reference_id)
    """

    plt.figure(1)
    plot_poses(camera_poses)   
    frame_i = [0]
    frame_fig = plt.figure(2)
    def project_frame():
        i = camera_poses.keys()[frame_i[0]%(len(camera_poses))]
        
        #frame_fig.clear()
        ax = frame_fig.add_subplot(1,1,1)
        ax.clear()
        rvec, tvec = camera_poses[i]
        frame = marker_data[i]
        rendered = set()
        for marker in frame['markers']:
            mid = marker['id']
            if mid in rendered: continue
            rendered.add(mid)
            try:
                world = marker_points[marker['id']]
            except KeyError:
                continue
            est = cv2.projectPoints(world, rvec, tvec, cm, cd)[0].reshape(-1, 2)
            ax.plot(est[:,0], est[:,1])
            mv = marker['verts'].reshape(-1, 2)
            ax.plot(mv[:,0], mv[:,1])
        ax.set_xlim((0, camera['resolution'][0]))
        ax.set_ylim((camera['resolution'][1], 0))
        frame_i[0] += 1
        frame_fig.canvas.draw()
        
        
    #fig = plt.figure(2)
    frame_fig.canvas.new_timer(interval=100, callbacks=[(project_frame, [], {})]).start()

    plt.figure(0)
    plt.clf()
    ax = plt.gcf().add_subplot(111, projection='3d')
    
    #for marker_id, res in enumerate(tpositions[0]):
    #    ax.text(res[0,0], res[0,1], res[0,2], str(marker_id))
    #    ax.plot(res[:,0], res[:,1], res[:,2], color='green')
    #for marker_id, res in pointify(marker_poses).iteritems():
    for marker_id, res in marker_points.iteritems():
        ax.text(res[0,0], res[0,1], res[0,2], str(marker_id))
        ax.plot(res[:,0], res[:,1], res[:,2], color='red')
    axisEqual3D(ax)
    plt.show()
    return
    screens = defaultdict(dict)
    for i, markers in enumerate(marker_data):
        for marker in markers:
            screens[marker['id']][i] = marker['verts']
    max_screens = max((len(s) for s in screens.values()))
    #known_locations = {reference_id: mdict[reference_id]}
    #whitelist = [reference_id, 8, 44, 19, 27, 10, 7]
    #whitelist = [reference_id, 7, 4]
    #whitelist = [reference_id, 8, 19, 27]
    #whitelist = [reference_id, 8, 44]
    #whitelist = [reference_id, 44, 8, 5, 4, 28, 8]
    estimated_locations = {}
    for marker_id in screens:
            if marker_id == reference_id: continue
            #if marker_id not in whitelist: continue
            #if len(screens[marker_id]) < max_screens*frequency_threshold:
            #    continue
            #estimated_locations[marker_id] = mdict[marker_id]
            estimated_locations[marker_id] = initial_marker_pose()
    min_points = 4
    best_matches = defaultdict(lambda: np.inf)
    #good_locations = {reference_id: mdict[reference_id]}
    good_locations = {reference_id: initial_marker_pose()}
    
    
    for i in range(10):
        if i == 0:
            pose_locations = good_locations
        else:
            pose_locations = dict(good_locations, **estimated_locations)
        poses = estimate_poses(marker_data, pointify(pose_locations), cm, cd, min_points=min_points)
        min_points = min(len(good_locations), 3)*4
        new_estimated_locations, errors = estimate_marker_locations(marker_data, poses, estimated_locations, mdict, cm, cd)
        enhanced = 0
        besterr = np.inf; best_k = None
        for k, loc in new_estimated_locations.iteritems():
            frame_errors = pixerrors(errors[k])
            if len(frame_errors) == 0:
                continue
            coverage = (len(frame_errors)/len(mdict[k]))/len(marker_data)
            error = coverage*np.percentile(frame_errors, 75) + (1 - coverage)*error_threshold
            estimated_locations[k] = loc
            if k in good_locations.keys():
                if error >= error_threshold:
                    del good_locations[k]
                    del estimated_locations[k]
                    #estimated_locations[k] = loc
                else:
                    good_locations[k] = loc
                    estimated_locations[k] = loc
                continue
            if error < besterr:
                besterr = error
                best_k = k
            #best_matches[k] = errors[k]
            #enhanced += 1
        """
        if best_k is None:
            print "No markers left"
            break
        if besterr > error_threshold:
            print "Removing marker", best_k, "with error", besterr
            del estimated_locations[best_k]
        else:
            #del estimated_locations[best_k]
            good_locations.update({best_k: estimated_locations[best_k]})
        #del estimated_locations[best_k]
        """

        #all_locations = dict(known_locations, **estimated_locations)
        plt.figure(0)
        plt.clf()
        ax = plt.gcf().add_subplot(111, projection='3d')
        for marker_id, res in pointify(pose_locations).iteritems():
            ax.text(res[0,0], res[0,1], res[0,2], str(marker_id))
            ax.plot(res[:,0], res[:,1], res[:,2])
        axisEqual3D(ax)
        plt.figure(1)
        plt.clf()
        i, rots, trans = map(np.array, zip(*poses))
        plt.plot(i, (trans[:,0]), color='black')
        plt.pause(0.1)
        """
        for marker_id, loc in all_locations.iteritems():
            locs = {marker_id: loc}
            tposes = estimate_poses(marker_data, locs, cm, cd, min_points=4)
            i, rots, trans = map(np.array, zip(*tposes))
            plt.plot(i, (trans[:,0]))
        plt.show()
        """
        #if enhanced == 0:
        #    break
    pickle.dump(good_locations, open(out_file, 'w'))
    if visualize: 
        poses = estimate_poses(omarker_data, pointify(good_locations), cm, cd, min_points=min_points)
        i, rots, trans = map(np.array, zip(*poses))
        plt.figure(1)
        plt.clf()
        for d in range(3):
            plt.plot(i, (trans[:,d]), '.-')
        plt.show()

if __name__ == '__main__':
    import argh; argh.dispatch_command(test)
