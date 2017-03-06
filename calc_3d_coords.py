import numpy as np
from scipy import interpolate
import msgpack


def applyProjection(m, vec):
    m = m.reshape(16, 1)
    x = vec[0]
    y = vec[1]
    z = vec[2]
    d = 1 / (m[3]*x + m[7]*y + m[11]*z + m[15])
    x0 = (m[0]*x +m[4]*y +m[8]*z + m[12])*d
    y0 = (m[1]*x +m[5]*y +m[9]*z + m[13])*d
    z0  = (m[2]*x +m[6]*y +m[10]*z + m[14])*d
    return np.array([x0, y0, z0])


def projec3D(x,y, projM, invWorldM, height):

        pos0 = applyProjection(np.linalg.inv(projM), (x,y,0.0))
        pos0 = applyProjection(np.linalg.inv(invWorldM), pos0)

        pos1 = applyProjection(np.linalg.inv(projM), (x,y, 0.5))
        pos1 = applyProjection(np.linalg.inv(invWorldM), pos1)

        dir_vec = pos1 - pos0
        t = -((pos0[1] - height)/dir_vec[1])
        pos3d = pos0 + (dir_vec)*t

        return pos3d


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

if __name__ == '__main__':
    
    #path = 

    both = np.load(path + "screen_coords_binocular.npy")
    eye0 = np.load(path + "screen_coords_eye0.npy")
    eye1 = np.load(path + "screen_coords_eye1.npy")
    
    xbin_interp = interpolate.interp1d(both[:,2], both[:,0])
    ybin_interp = interpolate.interp1d(both[:,2], both[:,1])


    x0_interp = interpolate.interp1d(eye0[:,2], eye0[:,0])
    y0_interp = interpolate.interp1d(eye0[:,2], eye0[:,1])

    x1_interp = interpolate.interp1d(eye1[:,2], eye1[:,0])
    y1_interp = interpolate.interp1d(eye1[:,2], eye1[:,1])

    sim_0 = open(path + "koep.msgpack")
    sim_0 = list(msgpack.Unpacker(sim_0, encoding='utf-8'))

    bin_coords = []
    right_coords = []
    left_coords = []

    trial = 0

    for pt in sim_0:
        try:
            scenario = pt['data']['loadingScenario']
            trial += 1
            start = pt['time']
            print start
        except:
            pass
        try:
            time = pt['time']
            invWorldM = np.array(pt['data']['camera']['matrixWorldInverse']).reshape((4,4))
            projM = np.array(pt['data']['camera']['projectionMatrix']).reshape((4,4))

            x =  pt['data']['physics']['bodies'][1]['position']['x']
            y =  pt['data']['physics']['bodies'][1]['position']['z']
            
            i = find_nearest(both[:,2],time)
            nearest = both[i]
            dif = abs(time - nearest[2])

            i = find_nearest(eye0[:,2],time)
            nearest = eye0[i]
            dif0 = abs(time - nearest[2])

            i = find_nearest(eye1[:,2],time)
            nearest = eye1[i]
            dif1 = abs(time - nearest[2])

            height = -0.09

            #binocular
            xg = 2.0*(xbin_interp(time)/1920.0) - 1
            yg = -2.0*((1080 - ybin_interp(time))/1080.0) + 1
            binocular = projec3D(xg, yg, projM, invWorldM, height)
                        
            #eye0
            xg = 2.0*(x0_interp(time)/1920.0) - 1
            yg = -2.0*((1080 - y0_interp(time))/1080.0) + 1
            right = projec3D(xg, yg, projM, invWorldM, height)

            #eye1
            xg = 2.0*(x1_interp(time)/1920.0) - 1
            yg = -2.0*((1080 - y1_interp(time))/1080.0) + 1
            left = projec3D(xg, yg, projM, invWorldM, height)

            if pt['data']['telemetry']['steering'] != 0 and dif < 1/60.0:
                bin_coords.append([float(binocular[0]), float(binocular[2]), y, x, time, trial])
            if pt['data']['telemetry']['steering'] != 0 and dif0 < 1/60.0:
                right_coords.append([float(right[0]), float(right[2]), y, x, time, trial])
            if pt['data']['telemetry']['steering'] != 0 and dif1 < 1/60.0:
                left_coords.append([float(left[0]), float(left[2]), y, x, time, trial])

                
        except:
            pass


    bin_coords = np.array(bin_coords)
    right_coords = np.array(right_coords)
    left_coords = np.array(left_coords)

    #print new_coords.shape
    np.save(path + "world_coords_both.npy", bin_coords)
    np.save(path + "world_coords_eye0.npy", right_coords)
    np.save(path + "world_coords_eye1.npy", left_coords)




    

    

