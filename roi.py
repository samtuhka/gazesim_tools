

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy import stats, signal
import os
import pickle



r = (7.5 ) 

xr = r * math.cos(math.pi/4.0)
yr = r * math.sin(math.pi/4.0)
a = 1. 


def lines(x,y):
    r = (2.5 )
    l = (r**2*2)**0.5 / 2.
    return np.array([(x - l, y),(x, y + l),(x + l, y),(x, y - l), (x - l, y)])

sq1 = lines(r,0)
sq2 = lines(-r,0)
sq3 = lines(xr,-yr)
sq4 = lines(-xr, -yr)

plt.rc('font', size=18)
plt.rc('xtick', labelsize=14)   
plt.rc('ytick', labelsize=14)

fig = plt.gcf()
r = 2.75
print r

fig.gca().add_patch(patches.Circle((0,0),r, color='black', fill=False))

roi = [[-4. / a, -2.5/a], [-4. / a, 7 / a], [4. / a, 7 / a], [4. / a, -2.5/a]]
roi1 = [[2.5 / a, -2.5/a], [2.5 / a, -4. / a], [-2.5 / a, -4. / a], [-2.5 / a, -2.5/a]]

roi = roi1 + roi
roi = np.array(roi)
#roi[:,0] += 960
#roi[:,1] += 540

roi1 = np.array(roi1)
#roi1[:,0] += 960
#roi1[:,1] += 540

plt.fill(roi[:,0],roi[:,1], color = "red", alpha = 0.5)
#plt.fill(roi1[:,0],roi1[:,1], color = "red", alpha = 0.5)


plt.plot(sq1[:,0], sq1[:,1], color = "black")
plt.plot(sq2[:,0], sq2[:,1], color = "black")
plt.plot(sq3[:,0], sq3[:,1], color = "black")
plt.plot(sq4[:,0], sq4[:,1], color = "black")
plt.ylim(-1080/27.428571429*0.5, 1080/27.428571429*0.5)
plt.xlim(-35,35)
plt.xlabel("x-koordinaatti (asteina)")
plt.ylabel("y-koordinaatti (asteina)")
plt.legend()
plt.show()






#            if len(xdist[xdist >= 4.]) < 3 and (len(ydist[ydist <= -2.5]) < 3 or len(xdist[xdist >= 2.5] < 3)) and len(ydist[ydist <= -4.0]) < 3: #np.median(dist_alt)*ang_px <= 10.0:
