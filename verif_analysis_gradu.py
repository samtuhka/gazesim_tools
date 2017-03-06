import numpy as np
import os
import matplotlib.pyplot as plt
from file_methods import load_object
from verif import verification
import math

cal_multi = []
cal_bin = []
cal_3D = []

orig_datas = []

def cal_stuff(pt_cloud, fix = False):
    res = np.sqrt(1280**2+720**2)
    field_of_view = 90
    px_per_degree = res/field_of_view
    
    error_h =  np.abs(pt_cloud['hor'][2] - pt_cloud['hor_ref'][2])
    error_h /= px_per_degree


    error_v = np.abs(pt_cloud['ver'][2] - pt_cloud['ver_ref'][2])
    error_v /= px_per_degree

    accuracy = (error_h**2 + error_v**2)**0.5

    accuracy_h = np.mean(error_h[accuracy < 5.])
    accuracy_v = np.mean(error_v[accuracy < 5.])

    error_h = error_h[accuracy < 5.]
    error_h = np.sort(error_h)

    error_v = error_v[accuracy < 5.]
    error_v = np.sort(error_v)

    high_h = error_h[int(len(error_h)*0.975)]
    low_h = error_h[int(len(error_h)*0.025)]


    high_v = error_v[int(len(error_v)*0.95)]
    low_v = error_v[int(len(error_v)*0.05)]

    new_hor = [0, 1, pt_cloud['hor'][2] - accuracy_h*px_per_degree*np.sign(np.mean(pt_cloud['hor'][2] - pt_cloud['hor_ref'][2]))]
    new_ver = [0, 1, pt_cloud['ver'][2] - accuracy_v*px_per_degree*np.sign(np.mean(pt_cloud['ver'][2] - pt_cloud['ver_ref'][2]))]
    new_cloud = {'hor':  new_hor, 'ver': new_ver, 'hor_ref': pt_cloud['hor_ref'], 'ver_ref': pt_cloud['ver_ref']}
    #if fix == False:
    #    print "first:", accuracy_h, accuracy_v
    #    cal_stuff(new_cloud, True)
    #else:
    #    print "second:", accuracy_h, accuracy_v
    
    #if len(error_h) > 100:
    #    print np.abs(accuracy_h - low), np.abs(high - accuracy_h), np.std(error_h, ddof = 1)*1.96

    #print "acc h:", accuracy_h, "std:", np.std(error_h[accuracy < 5.], ddof = 1)
    return np.std(error_h, ddof = 1), np.std(error_v, ddof = 1), high_h, low_h, high_v, low_v
    

rootdir = '/media/samtuhka/2b897bc1-d3af-4da7-b144-936442405274/SamuelKokeet/kokeet/'
rootdir = path = "/media/samtuhka/Seagate Expansion Drive/Mittaukset/"

stds = []
for dirs in os.walk(rootdir):
    path = str(dirs[0]) + "/"
    if os.path.exists(dirs[0] + "/verif_results_sim2"):
        data = load_object(dirs[0] + "/verif_results_sim2")
            
        if np.array(data['binocular']).shape != (3,5):
            data['binocular'] = data['binocular'][1:]
        print np.array(data['binocular']).shape
        cal_bin.append(data['binocular'])
    
        orig_data = load_object(dirs[0] + "/verif_data_sim2")
        #x,y, high_h, low_h, high_v, low_v = cal_stuff(orig_data['binocular'])
        #stds.append([x,y, high_h, low_h, high_v, low_v])
stds = np.array(stds)
cal_bin = np.array(cal_bin)
subjects = range(1,len(cal_bin)+ 1)
n = 23
#print np.mean(cal_bin[:,2,1] - cal_bin[:,1,1]), np.std(cal_bin[:,2,1], ddof = 1)
#print np.mean(cal_multi[:,2,1] - cal_multi[:,1,1]), np.std(cal_multi[:,2,1], ddof = 1)

#print np.mean(cal_bin[:,2,2] - cal_bin[:,1,2]), np.std(cal_bin[:,2,2], ddof = 1)
#print np.mean(cal_multi[:,2,2] - cal_multi[:,1,2]), np.std(cal_multi[:,2,2], ddof = 1)


print np.mean(cal_bin[:,1,1]), np.std(cal_bin[:,1,1], ddof = 1), n
print np.mean(cal_bin[cal_bin[:,2,1] > 0][:,2,1]), np.std(cal_bin[cal_bin[:,2,1] > 0][:,2,1], ddof = 1), n



print np.mean(cal_bin[:,1,2]), np.std(cal_bin[:,1,2], ddof = 1), n
print np.mean(cal_bin[cal_bin[:,2,2] > 0][:,2,2]), np.std(cal_bin[cal_bin[:,2,2] > 0][:,2,2], ddof = 1), n
#print np.mean(cal_multi[:,1,2]), np.std(cal_multi[:,1,2], ddof = 1), n
#cal_3D = cal_3D[cal_3D[:,1,2] > 0]
#print np.mean(cal_3D[:,1,2]), np.std(cal_3D[:,1,2], ddof = 1), n

plt.figure("horizontal")
plt.title("Horizontal")
plt.axhline(y=0, xmin=0, xmax=25, linewidth=2, color = 'k')
plt.plot(subjects,cal_bin[:,2,3] , '--o', label = "horizontal error")
plt.plot(subjects,-cal_bin[:,2,4] , '--o', label = "vertical error")

#plt.plot(subjects,cal_bin[:,2,3], '--o', label = "binocular hor")
#plt.plot(subjects, cal_multi[:,1,1], '--o', label = "multivariate mode")
#plt.plot(subjects, cal_3D[:,1,1], '--o',  label = "3d mode")
plt.xlim(0,25)
plt.ylim(-4,4)
plt.ylabel("mean accuracy")
plt.xlabel("subject id")
plt.legend()
plt.show()

#plt.figure("vertical")
#plt.axhline(y=2.0, xmin=0, xmax=25, linewidth=2, color = 'k')
#plt.plot(subjects,cal_bin[:,2,2] , '--o', label = "binocular")
#plt.plot(subjects, cal_multi[:,2,2], '--o', label = "multivariate")
#plt.plot(subjects, cal_3D[:,2,2], '--o', label = "3D")
#plt.legend()

#print np.mean(cal_bin[:,2,0]), np.std(cal_bin[:,2,0], ddof = 1)
#print np.mean(cal_multi[:,2,0]), np.std(cal_multi[:,2,0], ddof = 1)
#print np.mean(cal_3D[cal_3D[:,2,1] > 0][:,2,0]), np.std(cal_3D[cal_3D[:,2,1] > 0][:,2,0], ddof = 1)

cal_adjusted = []
error_x_high = []
error_x_low = []
error_y_high = []
error_y_low = []
for a,b,c, subject in zip(cal_bin, cal_bin, cal_3D, subjects):
    if  not c[2,1] > 0:
        c[2,1] = 100
        c[2,2] = 100
    hor = a[2,1]*np.sign(a[2,3])
    hor_std = stds[subject -1, 0]
    error_x_high.append(stds[subject - 1, 2])
    error_x_low.append(stds[subject - 1, 3])
    ver = min(a[2,2], c[2,2])
    if ver == a[2,2]:
        ver *=  np.sign(a[2,4])
        ver_std = stds[subject -1, 1]
        error_y_high.append(stds[subject - 1, 4])
        error_y_low.append(stds[subject - 1, 5])
    else:
        ver *=  np.sign(c[2,4])
        ver_std = stds3d[subject -1, 1]
        error_y_high.append(stds3d[subject - 1, 4])
        error_y_low.append(stds3d[subject - 1, 5])

    acc = (hor**2 + ver**2)**0.5
    cal_adjusted.append([hor, ver, acc, hor_std, ver_std])
cal_adjusted = np.array(cal_adjusted)


print np.mean(cal_adjusted[:,2]), np.std(cal_adjusted[:,2], ddof = 1)
#plt.figure()
#plt.scatter(cal_adjusted[:,0], cal_adjusted[:,1])

#print len(cal_bin[cal_bin[:,2,0] <= 2.0])
#print len(cal_multi[cal_multi[:,2,0] <= 2.0])
plt.figure()
#plt.plot(subjects, cal_adjusted[:,0], '--o', label = "horizontal")
#plt.plot(subjects, cal_adjusted[:,1], '--o', label = "vertical")
#plt.plot(subjects, cal_adjusted[:,0], '--o', color = "red", label = "accuracy hor")
#plt.plot(subjects, cal_adjusted[:,1], '--o', label = "accuracy ver")
#plt.plot(subjects, cal_adjusted[:,2], '--o', label = "accuracy")

error_x_low = np.array(error_x_low)
error_x_high = np.array(error_x_high)

#plt.errorbar(subjects, cal_adjusted[:,0], yerr=[np.abs(cal_adjusted[:,0] - error_x_low), np.abs(error_x_high - cal_adjusted[:,0])], fmt='--o', color = "green", label = "horizontal")
plt.errorbar(subjects, cal_adjusted[:,1], yerr=cal_adjusted[:,4], color = "green",  fmt='--o', label = "vertical")
#plt.fill_between(subjects, error_x_low, error_x_high)
plt.axhline(y=2.0, xmin=0, xmax=25, linewidth=2, color = 'k')
plt.axhline(y=-2.0, xmin=0, xmax=25, linewidth=2, color = 'k')
plt.axhline(y=0.0, xmin=0, xmax=25, linewidth=1, color = 'k', alpha = 0.5)
plt.legend()


error_y_low = np.array(error_y_low)
error_y_high = np.array(error_y_high)

plt.figure()
#plt.errorbar(subjects, cal_adjusted[:,1], yerr=[np.abs(cal_adjusted[:,1] - error_y_low), np.abs(error_y_high - cal_adjusted[:,1])], fmt='--o', color = "green", label = "vertical")
plt.errorbar(subjects, cal_adjusted[:,0], yerr=cal_adjusted[:,3], color = "green", fmt='--o', label = "horizontal")
plt.axhline(y=2.0, xmin=0, xmax=25, linewidth=2, color = 'k')
plt.axhline(y=-2.0, xmin=0, xmax=25, linewidth=2, color = 'k')
plt.axhline(y=0.0, xmin=0, xmax=25, linewidth=1, color = 'k', alpha = 0.5)
plt.legend()


#plt.plot(subjects, cal_bin[:,2,0], '--o', label = "bin accuracy")
#plt.plot(subjects, cal_multi[:,2,0], '--o', label = "multi accuracy")


#plt.plot(subjects,cal_bin[:,2,1] , '-o', label = "binocular")
#plt.plot(subjects, cal_multi[:,2,1], '-o', label = "multivariate")
#plt.plot(subjects, cal_3D[:,2,1], '-o', color = "blue", label = "3D")

plt.show()


"""
plt.figure("accuracy")
plt.plot(subjects,cal_bin[:,2,0], '-o', color = "green", label = "binocular")
#plt.plot(subjects, cal_multi[:,2,0], '-o', color = "red", label = "multivariate")
plt.legend()

plt.figure("vertical")
plt.plot(subjects,cal_bin[:,2,2], '-o', color = "green", label = "binocular")
plt.plot(subjects, cal_multi[:,2,2], '-o', color = "red", label = "multivariate")
plt.plot(subjects, cal_3D[:,2,2], '-o', color = "blue", label = "3D")
plt.legend()

plt.figure("horizontal")
plt.plot(subjects,cal_bin[:,2,1] , '-o', color = "green", label = "binocular")
plt.plot(subjects, cal_multi[:,2,1], '-o', color = "red", label = "multivariate")
plt.plot(subjects, cal_3D[:,2,1], '-o', color = "blue", label = "3D")
plt.legend()


plt.show()
"""
