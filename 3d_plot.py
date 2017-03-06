import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

path = "/home/samtuhka/pupil/recordings/2016_06_21/009/"

data = np.load(path + "world_coords_both.npy")
data0 = np.load(path + "world_coords_eye0.npy")
data1 = np.load(path + "world_coords_eye1.npy")

gaze = np.load(path + "screen_coords_binocular.npy")

trial = 4

lokaatiot = np.genfromtxt("lokaatiot", delimiter = " ", names = True)
valid = data[data[:,5] == trial]
valid0 = data0[data0[:,5] == trial]
valid1 = data1[data1[:,5] == trial]

#plt.figure()
#plt.plot(data[:,1], data[:,3])

alku = valid[0,4]
loppu = valid[-1,4]

gaze_v = gaze[gaze[:,2] > alku]
gaze_v = gaze_v[gaze_v[:,2] < loppu]

#gaze_y_vel = np.gradient(gaze_v[:,1])/(np.gradient(gaze_v[:,2]))
#gaze_y_vel = (gaze_y_vel / 1080.0) * (70 / (16./9.))

gaze_velocity = []
for gp0,gp1 in zip(gaze_v[:-1],gaze_v[1:]):
     dist_x = (gp1[0] - gp0[0]) / 1920.0 * 70.0
     dist_y = (gp1[1] - gp0[1]) / 1080.0 *  (70.0 * (9/16.0))
     dt = (gp1[2] - gp0[2])
     vel_x = dist_x / dt
     vel_y = dist_y / dt
     if dt > 0:
          gaze_velocity.append([gp1[2], vel_x, vel_y])
gaze_velocity = np.array(gaze_velocity)

plt.figure("vel")
plt.plot(gaze_velocity[:,0],gaze_velocity[:,2] , '-', color = "green")

plt.figure("vel hist")
plt.hist(gaze_velocity[:,2], 1000, normed=1, facecolor='green', alpha=0.75, range = (-150, 150))

plt.figure('x')
plt.plot(gaze_v[:,2], gaze_v[:,0], '.')

plt.figure('y')
plt.plot(gaze_v[:,2], gaze_v[:,1], '.')

plt.figure()
fig = plt.gcf()

rx = 50

if True:
     for i in range(45):
          t = (i/45.0)*math.pi
          x = rx*math.sin(t)
          y = rx*math.cos(t)
          #fig.gca().add_patch(patches.Circle((x,y),1, color='red', fill=False))

     #for i in range(20):
     #     x = (i/20.0)*1000

     fig.gca().add_patch(patches.Circle((0,0),rx-1.75, color='black', fill=False))
     fig.gca().add_patch(patches.Circle((0,0),rx+1.75, color='black', fill=False))


     #plt.hist2d(validit[:,0], validit[:,1], bins=2000, range = [[-200, 200],[-200,200]])

     plt.xlim(-200, 200)
     plt.ylim(-200, 200)
else:
     rx = 50
     s = (rx * 2 * math.pi * 15 / 360.0)
     for i in range(54):
          y = -600 + i*s
          fig.gca().add_patch(patches.Circle((0,y),0.1, color='red', fill=False))
     plt.axvline(x=-1.75, ymin=-600, ymax = 600, linewidth=2, color='black')
     plt.axvline(x=1.75, ymin=-600, ymax = 600, linewidth=2, color='black')
     plt.xlim(-10, 10)
     plt.ylim(-500, 500)

#plt.plot(valid0[:,0], valid0[:,1], '.', color = 'blue', label = "oikea")
#plt.plot(valid1[:,0], valid1[:,1], '.', color = 'green', label = "vasen")
plt.plot(valid[:,0], valid[:,1], '.', color = 'blue')
plt.plot(valid[:,3], valid[:,2], '.', color = 'green')


plt.legend()

     
plt.gca().invert_xaxis()
plt.axes().set_aspect('equal', 'datalim')
plt.show()
