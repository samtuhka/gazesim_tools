import numpy as np
from scipy import interpolate
import msgpack
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math

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


if __name__ == '__main__':
    path = "/home/samtuhka/2016_04_07/000/"
        
    new_coords = np.load(path + "new_coords.npy")

    data = new_coords[(new_coords[:,0] > -1000) & (new_coords[:,1] > -1000) ]
    data = data[(data[:,0] < 1000) & (data[:,1] < 1000)]


    x = (data[:,0] + 1000).astype(int)
    y = (data[:,1] + 1000).astype(int)

    print np.min(x), np.max(x)
    print np.min(y), np.max(y)

    r = (7.5 / 70.0) * 1920

    xr = r * math.cos(math.pi/4.0)
    yr = r * math.sin(math.pi/4.0)

    fig = plt.gcf()
    fig.gca().add_patch(patches.Circle((1000,1000),68.5, color='black', fill=False))

    fig.gca().add_patch(patches.Circle((1000 + r,1000),55, color='black', fill=False))
    fig.gca().add_patch(patches.Circle((1000 - r,1000),55, color='black', fill=False))

    fig.gca().add_patch(patches.Circle((1000 + xr,1000 - yr),55, color='black', fill=False))
    fig.gca().add_patch(patches.Circle((1000 - xr,1000 - yr),55, color='black', fill=False))

    plt.hist2d(x, y, bins=1000)
    plt.ylim(500, 1500)
    plt.xlim(0, 2000)
    plt.show()


    
    

    

