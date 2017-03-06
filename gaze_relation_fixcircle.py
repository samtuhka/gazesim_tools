import numpy as np
from scipy import interpolate, signal
import msgpack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pickle

def applyProjection(m, vec):
    m = m.reshape(16, 1)
    x = vec[0]
    y = vec[1]
    z = vec[2]
    d = 1 / (m[3]*x + m[7]*y + m[11]*z + m[15])
    x0 = (m[0]*x +m[4]*y +m[8]*z + m[12])*d
    y0 = (m[1]*x +m[5]*y +m[9]*z + m[13])*d
    z0  = (m[2]*x +m[6]*y +m[10]*z + m[14])*d
    return x0, y0, z0

def denormalize(pos, (width, height), flip_y=False):
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y

def project(data, i):
    try:
        invWorldM = np.array(data[i]['data']['camera']['matrixWorldInverse']).reshape((4,4))
        projM = np.array(data[i]['data']['camera']['projectionMatrix']).reshape((4,4))
        fut = 2
        x = data[i]['data']['prediction'][fut]['x']
        y = data[i]['data']['prediction'][fut]['y']
    except:
        return
    vec = (y, 0, x)
    player_pos = data[i]['data']['player']['position']
    
    m = np.dot(projM, invWorldM)
    vec_new = applyProjection(invWorldM, vec)
    vec_new = applyProjection(projM, vec_new)
    x = ((vec_new[0]+1)*0.5)
    y = ((vec_new[1]+1)*0.5)
    x, y = denormalize((x, y),(1920, 1080), False)
    return [x, y]


def main(path):
    both = np.load(path + "screen_coords_binocular.npy")
    #eye0 = np.load(path + "screen_coords_eye0.npy")
    #eye1 = np.load(path + "screen_coords_eye1.npy")
    timestamps = np.load(path + "world_timestamps.npy")
    timestamps2 = np.load(path + "world_timestamps_unix.npy")


    ts_interp = interpolate.interp1d(timestamps, timestamps2)
    both[:,2] = ts_interp(both[:,3])
    
    
    x_interp = interpolate.interp1d(both[:,2], both[:,0])
    y_interp = interpolate.interp1d(both[:,2], both[:,1])

    sim_0 = open(path + "sim_data.msgpack")
    sim_0 = list(msgpack.Unpacker(sim_0, encoding='utf-8'))

    new_coords = []
    data = []

    subject_info = {}
    
    trial = 0
    for pt in sim_0:
        try:
            key = pt['data']['formData'][0]['name']
            value = pt['data']['formData'][0]['value']

            if key == 'drivingDist' and key in subject_info:
                subject_info['drivingDistLife'] = value

            if key not in subject_info:
                subject_info[key] = value
        except:
            pass
        try:
            scenario = pt['data']['loadingScenario']
            trial += 1
            event = 0
            new_event = False
            if "Rev" in scenario:
                direction = -1
            else:
                direction = 1
        except:
            pass
        try:
            time = pt['time']
            
            x = pt['data']['prediction'][2]['x']
            y = pt['data']['prediction'][2]['y']


            probes = pt['data']['probes']
            target = -1
            for p in probes:
                t = p['visible']
                if t == 1:
                    target = p['index']                


            targetsPresent = pt['data']['targetScreen']
            if not new_event and targetsPresent:
                event += 1
                new_event = True
            if not targetsPresent:
                new_event = False
            
            phase = pt['data']['player']['road_phase']['phase']
            direction = pt['data']['player']['road_phase']['direction']
            response = pt['data']['telemetry']['pYes']
            if not phase:
                phase = "Not Started"
            if response != True:
                response = False

            invWorldM = np.array(pt['data']['camera']['matrixWorldInverse']).reshape((4,4))
            projM = np.array(pt['data']['camera']['projectionMatrix']).reshape((4,4))

            vec = (y, 0, x)

            m = np.dot(projM, invWorldM)
            vec_new = applyProjection(invWorldM, vec)
            vec_new = applyProjection(projM, vec_new)
            x = ((vec_new[0]+1)*0.5)
            y = ((vec_new[1]+1)*0.5)
            x, y = denormalize((x, y),(1920, 1080), False)
            speed = pt['data']['player']['speed']

            gx = x_interp(time)
            gy = y_interp(time)
            #print phase
            data.append((float(x),float(y),float(gx), float(gy), float(speed), bool(targetsPresent), str(phase), str(direction), bool(response), int(trial), str(scenario), int(event), float(time), target))
            if speed > 0:
                new_coords.append([gx - x, gy - y, time, x, y])
        except:
            pass
    x = np.dtype([('x', 'f4'), ('y', 'f4'),('gx', 'f4'), ('gy', 'f4'), ('speed','f4'), ('present', 'b'), ('phase', '<U10'), ('direction', '<U10'), ('response', 'b'), ('trial', 'i4'), ('scenario', '<U10'),('event', 'i4'), ('time', 'f8'), ('target', 'i4')])
    
    data = np.array(data, dtype = x)
    np.save(path + "sync_data_new.npy", data)
    pickle.dump(subject_info, open(path + "/subject_info", 'w'))
    #new_coords = np.array(new_coords)
    #np.save(path + "new_coords.npy", new_coords)

    


if __name__ == '__main__':

    #main(path)
    
    rootdir = path = "/media/samtuhka/Seagate Expansion Drive/Mittaukset/"
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        if os.path.exists(path + "screen_coords_binocular_new.npy") and  os.path.exists(path + "sim_data.msgpack"):
            main(path)
            print path



    

    

