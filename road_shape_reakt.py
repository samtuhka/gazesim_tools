# -*- coding: utf-8 -*-
import msgpack
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import math
import numpy as np
from scipy.misc import comb
import matplotlib.patches as patches

def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=3000):

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals.tolist(), yvals.tolist()


from matplotlib import pyplot as plt


#this is awful. please don't look at it!
def getPath(adj):

    rX = 155.6181665787421
    rY = rX
    s = 177.77777777777777
    terrainSize = 4500

    c = 0.5522847498307933984022516322796
    ox = rX * c
    oy = rY * c
    dy = -2*rY
    k = (terrainSize/2 - rX - 1.75) - 500

    pathx = []
    pathy = []
    cpathx = []
    cpathy = []

    pathx1 = []
    pathy1 = []

    for i in range(0,6):
        
        rX -= adj
        rY -= adj
        xvals1, yvals1 = bezier_curve([(0, rX + k), (c*rY,rX + k), (rY, c*rX + k), (rY, 0 + k)], nTimes=3000)
        xvals2, yvals2 = bezier_curve([(rY, 0 + k), (rY, -c*rX + k), (c*rY, -rX + k), (0, -rX + k)], nTimes=3000)

        rX -= 1.75
        rY -= 1.75
        xvals11, yvals11 = bezier_curve([(0, rX + k), (c*rY,rX + k), (rY, c*rX + k), (rY, 0 + k)], nTimes=3000)
        xvals21, yvals21 = bezier_curve([(rY, 0 + k), (rY, -c*rX + k), (c*rY, -rX + k), (0, -rX + k)], nTimes=3000)
        rX += 1.75
        rY += 1.75

        rX += 1.75
        rY += 1.75
        xvals111, yvals111 = bezier_curve([(0, rX + k), (c*rY,rX + k), (rY, c*rX + k), (rY, 0 + k)], nTimes=3000)
        xvals211, yvals211 = bezier_curve([(rY, 0 + k), (rY, -c*rX + k), (c*rY, -rX + k), (0, -rX + k)], nTimes=3000)
        rX -= 1.75
        rY -= 1.75
        
        x = np.linspace(0, -s, 3000).tolist()
        y = np.linspace(-rX + k, -rX + k,3000).tolist()

        x11 = np.linspace(0, -s, 3000).tolist()
        y11 = np.linspace(-rX + k + 1.75, -rX + k + 1.75,3000).tolist()


        x12 = np.linspace(0, -s, 3000).tolist()
        y12 = np.linspace(-rX + k - 1.75, -rX + k - 1.75,3000).tolist()

        xvals3, yvals3 = bezier_curve([(-s, rX + dy + k), (-c*rY - s, rX + dy + k), (-rY - s, c*rX + dy + k), (-rY - s, 0 + dy + k)], nTimes=3000)
        xvals4, yvals4 = bezier_curve([(-rY - s, 0 + dy + k), (-rY - s, -c*rX + dy + k), (-c*rY - s, -rX + dy + k), (-s, -rX + dy + k)], nTimes=3000)

        rX -= 1.75
        rY -= 1.75
        xvals31, yvals31 = bezier_curve([(-s, rX + dy + k), (-c*rY - s, rX + dy + k), (-rY - s, c*rX + dy + k), (-rY - s, 0 + dy + k)], nTimes=3000)
        xvals41, yvals41 = bezier_curve([(-rY - s, 0 + dy + k), (-rY - s, -c*rX + dy + k), (-c*rY - s, -rX + dy + k), (-s, -rX + dy + k)], nTimes=3000)
        rX += 1.75
        rY += 1.75

        rX += 1.75
        rY += 1.75
        xvals311, yvals311 = bezier_curve([(-s, rX + dy + k), (-c*rY - s, rX + dy + k), (-rY - s, c*rX + dy + k), (-rY - s, 0 + dy + k)], nTimes=3000)
        xvals411, yvals411 = bezier_curve([(-rY - s, 0 + dy + k), (-rY - s, -c*rX + dy + k), (-c*rY - s, -rX + dy + k), (-s, -rX + dy + k)], nTimes=3000)
        rX -= 1.75
        rY -= 1.75

        x2 = np.linspace(-s, 0, 3000).tolist()
        y2 = np.linspace(rX + 2*dy + k, rX + 2*dy + k, 3000).tolist()

        x21 = np.linspace(-s, 0, 3000).tolist()
        y21 = np.linspace(rX + 2*dy + k + 1.75, rX + 2*dy + k + 1.75, 3000).tolist()

        x22 = np.linspace(-s, 0, 3000).tolist()
        y22 = np.linspace(rX + 2*dy + k - 1.75, rX + 2*dy + k - 1.75, 3000).tolist()

        xvals1 = list(reversed(xvals1))
        yvals1 = list(reversed(yvals1))
        xvals2 = list(reversed(xvals2))
        yvals2 = list(reversed(yvals2))
        xvals3 = list(reversed(xvals3))
        yvals3 = list(reversed(yvals3))
        xvals4 = list(reversed(xvals4))
        yvals4 = list(reversed(yvals4))

        xvals11 = list(reversed(xvals11))
        yvals11 = list(reversed(yvals11))
        xvals21 = list(reversed(xvals21))
        yvals21 = list(reversed(yvals21))
        xvals31 = list(reversed(xvals31))
        yvals31 = list(reversed(yvals31))
        xvals41 = list(reversed(xvals41))
        yvals41 = list(reversed(yvals41))

        xvals111 = list(reversed(xvals111))
        yvals111 = list(reversed(yvals111))
        xvals211 = list(reversed(xvals211))
        yvals211 = list(reversed(yvals211))
        xvals311 = list(reversed(xvals311))
        yvals311 = list(reversed(yvals311))
        xvals411 = list(reversed(xvals411))
        yvals411 = list(reversed(yvals411))

        xvals = xvals1 + xvals2 + x + xvals3 + xvals4 + x2 + xvals11 + xvals21 + xvals31 + xvals41 + x11 + x12 + x21 + x22 + xvals111 + xvals211 + xvals311 + xvals411
        yvals = yvals1 + yvals2 +  y + yvals3 + yvals4 + y2 + yvals11 + yvals21 + yvals31 + yvals41 + y11 + y12 + y21 + y22 + yvals111 + yvals211 + yvals311 + yvals411
        if i % 2 == 0:
            outerx = xvals111 + xvals211 + x11 + xvals31 + xvals41 + x22
            outery = yvals111 + yvals211 + y11 + yvals31 + yvals41 + y22


            outerx1 = xvals11 + xvals21 + x12 + xvals311 + xvals411 + x21
            outery1 = yvals11 + yvals21 + y12 + yvals311 + yvals411 + y21

        else:
            outerx = xvals11 + xvals21 + x12 + xvals311 + xvals411 + x21
            outery = yvals11 + yvals21 + y12 + yvals311 + yvals411 + y21

            outerx1 = xvals111 + xvals211 + x11 + xvals31 + xvals41 + x22
            outery1 = yvals111 + yvals211 + y11 + yvals31 + yvals41 + y22


        #outerx = xvals11 + xvals21 + xvals31 + xvals41 + x11 + x12 + x21 + x22 + xvals111 + xvals211 + xvals311 + xvals411
        #outery = yvals11 + yvals21 + yvals31 + yvals41 + y11 + y12 + y21 + y22 + yvals111 + yvals211 + yvals311 + yvals411

        centx = xvals1 + xvals2 + x + xvals3 + xvals4 + x2 
        centy = yvals1 + yvals2 +  y + yvals3 + yvals4 + y2
        
        k += 2*dy
        pathx += outerx
        pathy += outery

        pathx1 += outerx1
        pathy1 += outery1

        cpathx += centx
        cpathy += centy
    return pathx, pathy, cpathx, cpathy, pathx1, pathy1



def edges(x,y):
    nx = []
    ny = []
    nx1 = []
    ny1 = []
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        y1 = y[i]
        y2 = y[i + 1]
        dx = x2 - x1
        dy = y2 - y1

        if dx==0 or dy==0:
            continue

        dist = (dx**2 + dy**2)**0.5
        dx /= dist
        dy /= dist

        x10 = x1 + (5*dy)
        y10 = y1 - (5*dx)

        x20 = x1 - (5*dy)
        y20 = y1 + (5*dy)

        c = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        #print c
        
        nx.append(x10)
        ny.append(y10)
        nx1.append(x20)
        ny1.append(y20)
        #plt.plot([x1, x2], [y1, y2], linewidth = 10)
    return nx, ny, nx1, ny1


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

def get_screen_coords(x,y, pt):
    invWorldM = np.array(pt['data']['camera']['matrixWorldInverse']).reshape((4,4))
    projM = np.array(pt['data']['camera']['projectionMatrix']).reshape((4,4))
    vec = (y, 0, x)

    m = np.dot(projM, invWorldM)
    vec_new = applyProjection(invWorldM, vec)
    vec_new = applyProjection(projM, vec_new)
    x = ((vec_new[0]+1)*0.5)
    y = ((vec_new[1]+1)*0.5)
    x, y = denormalize((x, y),(1920, 1080), False)
    return [x,y]

    


x,y,cx,cy, x1, y1= getPath(0)

path = "/home/samtuhka/2016_04_07/000/"
sim_0 = open(path + "koe.msgpack")
sim_0 = list(msgpack.Unpacker(sim_0, encoding='utf-8'))


def getPt(p, stop, t = 3):
    trial = 0
    for i in range(len(sim_0)):
            try:
                scenario = pt['data']['loadingScenario']
                trial += 1
                straights = 0
            except:
                pass
            try:
                pt = sim_0[i]
                time = pt['time']
                phase = pt['data']['player']['road_phase']['phase']
                if phase == p:
                    straights += 1
                if phase == p and trial == t and straights > stop:
                    print scenario
                    return pt
            except:
                pass
            
def getScreenCoords(pt, xdata, ydata):
    screen_coords = []
    for x,y in zip(xdata,ydata):
        c = get_screen_coords(x,y,pt)
        if c[1] <= 540:
            screen_coords.append(c)
    return screen_coords

pt0 = getPt('cornering', 30)
pt1 = getPt('straight', 1230)

pt2 = getPt('cornering', 30, t = 5)
pt3 = getPt('straight', 1230, t = 5)


screen_coords = getScreenCoords(pt0, x, y)
center_screen_coords = getScreenCoords(pt0, x1, y1)

screen_coords1 = getScreenCoords(pt1, x, y)
center_screen_coords1= getScreenCoords(pt1, x1, y1)



fut = 2
x = pt0['data']['prediction'][fut]['x']
y = pt0['data']['prediction'][fut]['y']
fix_circle0 = get_screen_coords(x,y,pt0)

x = pt1['data']['prediction'][fut]['x']
y = pt1['data']['prediction'][fut]['y']
fix_circle1 = get_screen_coords(x,y,pt1)


x = pt2['data']['prediction'][fut]['x']
y = pt2['data']['prediction'][fut]['y']
fix_circle2 = get_screen_coords(x,y,pt2)

x = pt3['data']['prediction'][fut]['x']
y = pt3['data']['prediction'][fut]['y']
fix_circle3 = get_screen_coords(x,y,pt3)



screen_coords = np.array(screen_coords)
center_screen_coords = np.array(center_screen_coords)

screen_coords1 = np.array(screen_coords1)
center_screen_coords1 = np.array(center_screen_coords1)


r = (7.5 / 70.0) * 1920
xr = r * math.cos(math.pi/4.0)
yr = r * math.sin(math.pi/4.0)

ylim = 1080*0.8
xmin = 0
xlim = 1920*0.8
fix_r = 2.75/70*1920

plt.rc('font', size=18)
plt.rc('xtick', labelsize=14)   
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)


fig, axes = plt.subplots( 2, 2 )
fig.suptitle("Reaktioajat", fontsize = 20)

axes[1,0].add_patch(patches.Circle((fix_circle0[0],fix_circle0[1]),fix_r, color='black', linewidth = 2, fill=False))
def lines(x,y):
    r = (2.5 / 70.0) * 1920
    l = (r**2*2)**0.5 / 2.
    return np.array([(x - l, y),(x, y + l),(x + l, y),(x, y - l), (x - l, y)])

sq1 = lines(fix_circle0[0] + r,fix_circle0[1])
sq2 = lines(fix_circle0[0] - r,fix_circle0[01])
sq3 = lines(fix_circle0[0] + xr,fix_circle0[1] - yr)
sq4 = lines(fix_circle0[0] - xr,fix_circle0[1] - yr)

points = [sq1, sq2, sq3, sq4]
points = np.array(points)

axes[1,0].title.set_text(u'Ohjaustehtävä (mutka)')
axes[1,0].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "M1")
axes[1,0].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "M2")
axes[1,0].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "M3")
axes[1,0].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "M4")


axes[1,0].plot(center_screen_coords[:,0], center_screen_coords[:,1], '-', linewidth = 1, color = "blue")
axes[1,0].plot(screen_coords[:,0], screen_coords[:,1], '-', linewidth = 1, markersize=0.75, color = "blue")
data = np.load("ajoR.npy")

for i,p in enumerate(data):
    loc = points[i,1]
    result = p[1]
    axes[1,0].text(loc[0], loc[1] + 25, "%.0f" % (result*1000) + "ms",  horizontalalignment='center', fontsize = 18)

axes[1,0].set_ylim([0,ylim])
axes[1,0].set_xlim([xmin,xlim])


#new subplot
axes[0,0].title.set_text(u'Ohjaustehtävä (suora)')
axes[0,0].add_patch(patches.Circle((fix_circle1[0],fix_circle1[1]),fix_r, color='black', linewidth = 2, fill=False))
sq1 = lines(fix_circle1[0] + r,fix_circle1[1])
sq2 = lines(fix_circle1[0] - r,fix_circle1[01])
sq3 = lines(fix_circle1[0] + xr,fix_circle1[1] - yr)
sq4 = lines(fix_circle1[0] - xr,fix_circle1[1] - yr)

points = [sq1, sq2, sq3, sq4]
points = np.array(points)

    
axes[0,0].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "S1")
axes[0,0].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "S2")
axes[0,0].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "S3")
axes[0,0].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "S4")
axes[0,0].plot(center_screen_coords1[:,0], center_screen_coords1[:,1], '-', linewidth = 1, color = "blue")
axes[0,0].plot(screen_coords1[:,0], screen_coords1[:,1], '-', linewidth = 1, markersize=0.75, color = "blue")
data = np.load("ajoR.npy")

for i,p in enumerate(data):
    loc = points[i,1]
    result = p[0]
    axes[0,0].text(loc[0], loc[1] + 25, "%.0f" % (result*1000) + "ms", horizontalalignment='center', fontsize = 18)
    
axes[0,0].set_ylim([0,ylim])
axes[0,0].set_xlim([xmin,xlim])




#new subplot
axes[1,1].title.set_text(u'Verrokkitehtävä (mutka)')
axes[1,1].add_patch(patches.Circle((fix_circle2[0],fix_circle2[1]),fix_r, color='black', linewidth = 2,  fill=False))
sq1 = lines(fix_circle2[0] + r,fix_circle2[1])
sq2 = lines(fix_circle2[0] - r,fix_circle2[01])
sq3 = lines(fix_circle2[0] + xr,fix_circle2[1] - yr)
sq4 = lines(fix_circle2[0] - xr,fix_circle2[1] - yr)
points = [sq1, sq2, sq3, sq4]
points = np.array(points)
    
axes[1,1].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "M1")
axes[1,1].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "M2")
axes[1,1].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "M3")
axes[1,1].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "M4")
data = np.load("darkR.npy")


for i,p in enumerate(data):
    loc = points[i,1]
    result = p[1]
    axes[1,1].text(loc[0], loc[1] + 25, "%.0f" % (result*1000) + "ms",  horizontalalignment='center', fontsize = 18)
axes[1,1].set_ylim([0,ylim])
axes[1,1].set_xlim([xmin,xlim])

#new subplot
axes[0,1].title.set_text(u'Verrokkitehtävä (suora)')
axes[0,1].add_patch(patches.Circle((fix_circle3[0],fix_circle3[1]),fix_r, color='black', linewidth = 2, fill=False))
sq1 = lines(fix_circle3[0] + r,fix_circle3[1])
sq2 = lines(fix_circle3[0] - r,fix_circle3[01])
sq3 = lines(fix_circle3[0] + xr,fix_circle3[1] - yr)
sq4 = lines(fix_circle3[0] - xr,fix_circle3[1] - yr)
points = [sq1, sq2, sq3, sq4]
points = np.array(points)
    
axes[0,1].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "S1")
axes[0,1].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "S2")
axes[0,1].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "S3")
axes[0,1].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "S4")
data = np.load("darkR.npy")


for i,p in enumerate(data):
    loc = points[i,1]
    result = p[0]
    axes[0,1].text(loc[0], loc[1] + 25, "%.0f" % (result*1000) + "ms",   horizontalalignment='center', fontsize = 18)
axes[0,1].set_ylim([0,ylim])
axes[0,1].set_xlim([xmin,xlim])

plt.subplots_adjust(wspace=0, hspace=0)

axes[0,0].axis('off')
axes[0,1].axis('off')
axes[1,0].axis('off')
axes[1,1].axis('off')
axes[0,1].legend(loc = 1, bbox_to_anchor=(1, 1))
axes[1,1].legend(loc = 1, bbox_to_anchor=(1, 1))

plt.savefig('foo1.png', dpi = 300, bbox_inches='tight', pad_inches = 0)

plt.show()




#plt.plot(xvals1, yvals1)
#plt.plot(pathx, pathy)
#plt.plot(xvals2[:200], yvals2[:200])

#plt.plot(xvals2[:900], yvals2[:900])

#print xvals
#plt.plot(xpoints, ypoints, "ro")
#plt.show()

