# -*- coding: cp1252 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy import stats, signal
import os
import pickle
from pandas import DataFrame

def adjustFigAspect(fig,aspect=1):
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def target_location(phase, direction, target):
    #print phase
    if phase == "straight":
        pos = 0
    if phase == "approach":
        pos = 1
    if phase == "cornering" or phase == "cornerin":
        pos = 2
    if phase == "exit":
        pos = 3
        
    if direction == "right" and target == 0:
        target = 1
    elif direction == "right" and target == 1:
        target = 0
    elif direction == "right" and target == 2:
        target = 3
    elif direction == "right" and target == 3:
        target = 2

    return target, pos

    



def main(path, data):
    data = np.load(path + data)
    #print data
    new_data = []
    accepted = 0
    purged = 0

    purged_gazes = []

    total = np.zeros((4,4))
    hits = np.zeros((4,4))
    total_dark = np.zeros((4,4))
    hits_dark = np.zeros((4,4))
    no_event = 0
    false_positive = 0


    ang_px = 70. / 1920

    react_list = []
    react_list_dark = []
    for i in range(4):
        r = []
        d = []
        for j in range(4):
            r.append([])
            d.append([])
        react_list.append(r)
        react_list_dark.append(d)

    for trial in np.unique(data['trial']):
        trial = data[data['trial'] == trial]
        if trial['trial'][0] <= 2:
            continue
        trialGxVel = abs(np.diff(trial['gx'])) / np.diff(trial['time'])
        trialGyVel = abs(np.diff(trial['gy'])) / np.diff(trial['time'])
        #print np.nanmedian(trialGyVel)*ang_px
        for event in np.unique(trial['event']):
            event = trial[trial['event'] == event]

            reaction_time = 0
            response = event['response']
            response = np.where(response==1)
            if len(response[0]) > 0:
                response = response[0][0]
                reaction_time = event[response]['time']
            
            event = event[event['present'] == 1]
            event = event[event['target'] >= 0]

            if len(event) == 0:
                no_event += 1
                if reaction_time > 0:
                    false_positive += 1
                continue

            reaction_time -= event[0]['time']

            scenario = event['scenario'][0]
            phase = event['phase']
            phase = (stats.mode(phase)[0])[0]
            direction = event['direction']
            direction = (stats.mode(direction)[0])[0]
            target = event['target'][0]
            print direction

            t, p = target_location(phase, direction, target)

            if "dark" in scenario:
                total_dark[t, p] += 1
            else:
                total[t, p] += 1

            found = False
            xdist = np.array([1000,1000,1000])
            ydist = xdist
            if len(event) > 1:
                found = True
                xdist = abs(event['gx'] - event['x'])
                ydist = (event['gy'] - event['y'])
                xdist = np.array(xdist)*ang_px
                ydist = np.array(ydist)*ang_px


            if len(xdist[xdist >= 4.]) < 3 and (len(ydist[ydist <= -2.5]) < 3 or len(xdist[xdist >= 2.5] < 3)) and len(ydist[ydist <= -4.0]) < 3: #np.median(dist_alt)*ang_px <= 10.0:
                accepted += 1

                if reaction_time > 0 and "dark" in scenario:
                    hits_dark[t, p] += 1
                    react_list_dark[t][p].append(reaction_time)
                elif reaction_time > 0:
                    hits[t, p] += 1
                    react_list[t][p].append(reaction_time)

                for g in event:
                    hor = g['gx'] - g['x']
                    ver = g['gy'] - g['y']
                    new_data.append([hor, ver])
            elif found:
                purged += 1
                for g in event:
                    hor = g['gx'] - g['x']
                    ver = g['gy'] - g['y']
                    if abs(hor)*ang_px >= 4.0 or (ver*ang_px < -2.5 and abs(hor)*ang_px >= 3) or ver*ang_px < -4.0:
                        purged_gazes.append([hor, ver])


    #print false_positive*1./no_event
            
    print purged*1.0/(purged + accepted), accepted

    react = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            react[i,j] = np.median(np.array(react_list[i][j]))

    react_dark = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            react_dark[i,j] = np.median(np.array(react_list_dark[i][j]))

    corr = hits/total
    corr_d = hits_dark/total_dark

    x = np.arange(2)
    x2 = np.arange(2) + 2
    x3 = np.arange(4)

    x_labs = ["suora(ohjaus)","mutka(ohjaus)", "suora()", "mutka()"]


    corr = np.delete(corr, [1,3], 1)
    corr_d = np.delete(corr_d, [1,3], 1)
    """
    plt.figure("ajo")

    plt.plot(corr[0], 'o--', label = "ylin suun")
    plt.plot(corr[1], 'o--', label = "ylin ~suun")
    plt.plot(corr[2], 'o--', label = "alin suun")
    plt.plot(corr[3], 'o--', label = "alin ~suun")

    plt.plot(x2, corr_d[0], 'o--', color = "blue")
    plt.plot(x2, corr_d[1], 'o--', color = "green")
    plt.plot(x2, corr_d[2], 'o--', color = "red")
    plt.plot(x2, corr_d[3], 'o--', color = "cyan")

    plt.ylabel("corr%")
    plt.xticks(x3, x_labs)
    plt.xlim(-0.5, 4.5)
    plt.legend()


    react = np.delete(react, [1,3], 1)
    react_dark = np.delete(react_dark, [1,3], 1)
    plt.figure("ajo_reaktio")
    plt.plot(react[0], 'o--', label = "ylin suun")
    plt.plot(react[1], 'o--', label = "ylin ~suun")
    plt.plot(react[2], 'o--', label = "alin suun")
    plt.plot(react[3], 'o--', label = "alin ~suun")

    plt.plot(x2, react_dark[0], 'o--', color = "blue")
    plt.plot(x2, react_dark[1], 'o--', color = "green")
    plt.plot(x2, react_dark[2], 'o--', color = "red")
    plt.plot(x2, react_dark[3], 'o--', color = "cyan")


    plt.ylabel("react")
    plt.xticks(x3, x_labs)
    plt.xlim(-0.5, 4.5)
    plt.legend()
    """
    if accepted > 1:
        pass
        """
        plt.figure()
        new_data.append([-1000,-500])
        new_data.append([1000,500])
        data = np.array(new_data)
        #data = new_data[(new_data[:,0] > -1000) & (new_data[:,1] > -1000) ]
        #data = data[(data[:,0] < 1000.) & (data[:,1] < 1000.)]
        #print data[:,0]

        x = (data[:,0] + 1000).astype(int)
        y = (data[:,1] + 1000).astype(int)
        condition = (x <= 2000) & (y <= 1500) & (x >= 0) & (y >= 500)

        x = x[condition]
        y = y[condition]
        
        r = (7.5 / 70.0) * 1920

        xr = r * math.cos(math.pi/4.0)
        yr = r * math.sin(math.pi/4.0)
        
        fig = plt.gcf()
        fig.gca().add_patch(patches.Circle((1000,1000),68.5, color='black', fill=False))

        fig.gca().add_patch(patches.Circle((1000 + r,1000),55, color='black', fill=False))
        fig.gca().add_patch(patches.Circle((1000 - r,1000),55, color='black', fill=False))

        fig.gca().add_patch(patches.Circle((1000 + xr,1000 - yr),55, color='black', fill=False))
        fig.gca().add_patch(patches.Circle((1000 - xr,1000 - yr),55, color='black', fill=False))
        #plt.scatter(x,y)
        h = np.max(x) - np.min(x)
        w = np.max(y) - np.min(y)
        #bins = max(min(int((h*w) / 1000.), 4000), 100)
        #plt.scatter(x,y)
        plt.hist2d(x, y, bins = 100)
        plt.ylim(500, 1500)
        plt.xlim(0, 2000)
        """
    data_dict = {'hits': hits, 'total': total, 'hits_dark': hits_dark, 'total_dark': total_dark, 'react': react, 'react_dark': react_dark, 'newn_data': new_data, 'purged_gazes': purged_gazes, 'no_event': no_event, 'false_positive': false_positive}
    return hits, total, hits_dark, total_dark, react, react_dark, new_data, data_dict

def save(ind, data):
    path = "koehenkilot/" + str(ind)
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(data, open(path + "/data", 'w'))
    


if __name__ == '__main__':
    #main("/media/samtuhka/Seagate Expansion Drive/Mittaukset/2016_04_15/001/")
    rootdir = "/media/samtuhka/Seagate Expansion Drive/Mittaukset/"
    #rootdir = "G:/Mittaukset/"
    n = 0
    purged = []
    new_data = []

    purged_datas = ["2016_04_08/001", "2016_04_12/001", "2016_04_19/000"] # "2016_04_14/000", "2016_04_15/000"]
    
    perf = np.zeros((4,4))
    perf_dark = np.zeros((4,4))
    react = np.zeros((4,4))
    react_dark = np.zeros((4,4))

    perf_list = []
    perf_dark_list = []
    react_list = []
    react_dark_list = []

    purged = []
    purged_dark = []

    gaze_data = []
    fp = []

    rn = 0

    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"

        skip = False

        for p in purged_datas:
            p = rootdir + p
            p = os.path.normpath(p)
            if p in os.path.normpath(path):
                skip = True
        if os.path.exists(path + "sim_data.msgpack") and not skip:
            pass
    for i in range(23):
            n += 1

            path = "koehenkilot/" + str(n) + "/data" 
            data_dict = pickle.load(open(path, 'r'))
            hits = data_dict['hits']
            total = data_dict['total']
            hits_dark = data_dict['hits_dark']
            total_dark = data_dict['total_dark']
            print sum(total + total_dark)

            r = data_dict['react']
            r_d = data_dict['react_dark']
            gazes = data_dict['newn_data']

            falsep = data_dict['false_positive']
            fp.append(falsep)

            pg = data_dict['purged']
            pgd = data_dict['purged_dark']

            purged.append(pg)
            purged_dark.append(pgd)

            gaze_data.append(gazes)

            new_data += gazes
            perf += hits/total
            perf_dark += hits_dark/total_dark
            perf_list.append(hits/total)
            perf_dark_list.append(hits_dark/total_dark)
            react += r
            react_dark += r_d
            react_list.append(r)
            react_dark_list.append(r_d)

            #main(path, "sync_data.npy")
    #plt.show()

    fp = np.array(fp)
    
    fp = fp / 1200.
    print np.mean(fp), np.std(fp, ddof = 1)
    purged = np.array(purged)
    purged_dark = np.array(purged_dark)
    #print np.mean(purged), np.mean(purged_dark)
    #print stats.ttest_rel(purged, purged_dark)
    gaze_data = np.array(gaze_data)
    #print gaze_data[0]

    
    perf_list = np.array(perf_list)
    perf_dark_list = np.array(perf_dark_list)
    react_list = np.array(react_list)
    react_dark_list = np.array(react_dark_list)
    print perf_list.shape

    ylos = (perf_list[:,0,0] + perf_list[:,1,0]) / 2.
    alas = (perf_list[:,3,0] + perf_list[:,2,0]) / 2.
    ylosd = (perf_dark_list[:,0,0] + perf_dark_list[:,1,0]) / 2.
    alasd = (perf_dark_list[:,3,0] + perf_dark_list[:,2,0]) / 2.

    #plt.figure()
    #for x,y in zip(ylos,alas): plt.plot([x,y], alpha = 0.5)


    corr = perf / n
    corr_d = perf_dark / n
    react /= n
    react_dark /= n

    x = np.arange(2)
    x2 = np.arange(2) + 2
    x3 = np.arange(4)
    
    x_labs = ["suora(ohjaus)","mutka(ohjaus)", "suora()", "mutka()"]


    corr = np.delete(corr, [1,3], 1)
    corr_d = np.delete(corr_d, [1,3], 1)


    s = 0
    c = 2

    print stats.friedmanchisquare(react_list[:,0,s] - react_dark_list[:,0,s], react_list[:,1,s] - react_dark_list[:,1,s], react_list[:,2,s] - react_dark_list[:,2,s], react_list[:,3,s] - react_dark_list[:,3,s])
    print stats.friedmanchisquare(react_list[:,0,c] - react_dark_list[:,0,c], react_list[:,1,c] - react_dark_list[:,1,c], react_list[:,2,c] - react_dark_list[:,2,c], react_list[:,3,c] - react_dark_list[:,3,c])
    print "yle suoritustaso suora", stats.wilcoxon(react_list[:,0,s] + react_list[:,1,s] + react_list[:,2,s] + react_list[:,3,s],react_dark_list[:,0,s] + react_dark_list[:,1,s] + react_dark_list[:,2,s] + react_dark_list[:,3,s])
    print "yle suoritustaso mutka", stats.wilcoxon(react_list[:,0,c] + react_list[:,1,c] + react_list[:,2,c] + react_list[:,3,c],react_dark_list[:,0,c] + react_dark_list[:,1,c] + react_dark_list[:,2,c] + react_dark_list[:,3,c])


    print stats.friedmanchisquare(perf_list[:,0,s] - perf_dark_list[:,0,s], perf_list[:,1,s] - perf_dark_list[:,1,s], perf_list[:,2,s] - perf_dark_list[:,2,s], perf_list[:,3,s] - perf_dark_list[:,3,s])
    print stats.friedmanchisquare(perf_list[:,0,c] - perf_dark_list[:,0,c], perf_list[:,1,c] - perf_dark_list[:,1,c], perf_list[:,2,c] - perf_dark_list[:,2,c], perf_list[:,3,c] - perf_dark_list[:,3,c])


    print stats.friedmanchisquare(perf_list[:,0,s], perf_list[:,1,s], perf_list[:,2,s], perf_list[:,3,s])
    print stats.friedmanchisquare(perf_list[:,0,c], perf_list[:,1,c], perf_list[:,2,c], perf_list[:,3,c])

    print " ¨alin suun versus ylin suun suoralla:", stats.wilcoxon(perf_list[:,3,s] - perf_dark_list[:,3,s], perf_list[:,0,s] - perf_dark_list[:,0,s])


    #perf_list = np.log(perf_list)
    #perf_dark_list = np.log(perf_dark_list)

    react_list = np.log(react_list)
    react_dark_list = np.log(react_dark_list)
    print "alin suun versus alin ~suun suoralla:", stats.wilcoxon(perf_list[:,2,s] - perf_dark_list[:,2,s], perf_list[:,3,s] - perf_dark_list[:,3,s])
    print "alin suun versus ylin suun suoralla:", stats.wilcoxon(perf_list[:,2,s] - perf_dark_list[:,2,s], perf_list[:,0,s] - perf_dark_list[:,0,s])
    print "alin suun versus ylin ~suun suoralla:", stats.wilcoxon(perf_list[:,2,s] - perf_dark_list[:,2,s], perf_list[:,1,s] - perf_dark_list[:,1,s])


    print " ¨alin suun versus ylin suun suoralla:", stats.wilcoxon(perf_list[:,3,s] - perf_dark_list[:,3,s], perf_list[:,0,s] - perf_dark_list[:,0,s])
    print " ¨alin suun versus ylin ~suun suoralla:", stats.wilcoxon(perf_list[:,3,s] - perf_dark_list[:,3,s], perf_list[:,1,s] - perf_dark_list[:,1,s])


    print " ¨ylin suun versus ylin ~suun suoralla:", stats.wilcoxon(perf_list[:,0,s] - perf_dark_list[:,0,s], perf_list[:,1,s] - perf_dark_list[:,1,s])


    print "yle suoritustaso suora", stats.wilcoxon(perf_list[:,0,s] + perf_list[:,1,s] + perf_list[:,2,s] + perf_list[:,3,s],perf_dark_list[:,0,s] + perf_dark_list[:,1,s] + perf_dark_list[:,2,s] + perf_dark_list[:,3,s])
    print "yle suoritustaso mutka", stats.wilcoxon(perf_list[:,0,c] + perf_list[:,1,c] + perf_list[:,2,c] + perf_list[:,3,c],perf_dark_list[:,0,c] + perf_dark_list[:,1,c] + perf_dark_list[:,2,c] + perf_dark_list[:,3,c])
    print "yle suoritustaso suora", stats.wilcoxon(perf_list[:,0,s] + perf_list[:,1,s] + perf_list[:,2,s] + perf_list[:,3,s] + perf_list[:,0,c] + perf_list[:,1,c] + perf_list[:,2,c] + perf_list[:,3,c],perf_dark_list[:,0,s] + perf_dark_list[:,1,s] + perf_dark_list[:,2,s] + perf_dark_list[:,3,s] + perf_dark_list[:,0,c] + perf_dark_list[:,1,c] + perf_dark_list[:,2,c] + perf_dark_list[:,3,c])

    #w,t = stats.wilcoxon(perf_list[:,2,s] - perf_dark_list[:,2,s], perf_list[:,3,s] - perf_dark_list[:,3,s])
    #print w/rank_sum
    
    ylemmat = (perf_list[:,0,s] - perf_dark_list[:,0,s] + perf_list[:,1,s] - perf_dark_list[:,1,s]) / 2.0
    alemmat = (perf_list[:,3,s] - perf_dark_list[:,3,s] + perf_list[:,2,s] - perf_dark_list[:,2,s]) / 2.0
    print "ylemmat vs alemmat ka norm suoralla:", stats.wilcoxon(ylemmat, alemmat)

    ylemmat = (perf_list[:,0,c]  - perf_dark_list[:,0,c] + perf_list[:,1,c] -  perf_dark_list[:,1,c]) / 2.0
    alemmat = (perf_list[:,3,c] - perf_dark_list[:,3,c] + perf_list[:,2,c] -  perf_dark_list[:,2,c]) / 2.0

    i = 0
    #for a,y in zip(ylemmat, alemmat):
    #    plt.plot([a,y], label = i)
    #    i += 1
    #    plt.legend()
    #plt.show()
    print "ylemmat vs alemmat ka norm mutkassa:", stats.wilcoxon(ylemmat, alemmat)



    ylemmat = (perf_list[:,0,c]  + perf_list[:,1,c]) / 2.0
    alemmat = (perf_list[:,3,c]  + perf_list[:,2,c]) / 2.0
    print "ylemmat vs alemmat ka mutkassa:", stats.wilcoxon(ylemmat, alemmat)


    ylemmat = (react_list[:,0,s] + react_list[:,1,s]) / 2.0
    alemmat = (react_list[:,3,s] + react_list[:,2,s]) / 2.0
    print "ylemmat vs alemmat ka reaktio suoralla:", stats.wilcoxon(ylemmat, alemmat)

    ylemmat = (react_list[:,0,c] + react_list[:,1,c]) / 2.0
    alemmat = (react_list[:,3,c] + react_list[:,2,c]) / 2.0
    print "ylemmat vs alemmat ka reaktio mutkassa:", stats.wilcoxon(ylemmat, alemmat)


    ylemmat = (react_dark_list[:,0,s] + react_dark_list[:,1,s]) / 2.0
    alemmat = (react_dark_list[:,3,s] + react_dark_list[:,2,s]) / 2.0
    print "ylemmat vs alemmat ka reaktio suoralla kontrolli:", stats.wilcoxon(ylemmat, alemmat)

    ylemmat = (react_dark_list[:,0,c] + react_dark_list[:,1,c]) / 2.0
    alemmat = (react_dark_list[:,3,c] + react_dark_list[:,2,c]) / 2.0
    print "ylemmat vs alemmat ka reaktio mutkassa kontrolli:", stats.wilcoxon(ylemmat, alemmat)


    suora = (perf_list[:,0,s] + perf_list[:,1,s] + perf_list[:,2,s] + perf_list[:,3,s]) 
    mutka = (perf_list[:,0,c] + perf_list[:,1,c] + perf_list[:,2,c] + perf_list[:,3,c]) 
    print "suora vs mutk:", stats.wilcoxon(suora, mutka)



    
    suora = (perf_list[:,0,s] - perf_dark_list[:,0,s] + perf_list[:,1,s] - perf_dark_list[:,1,s] + perf_list[:,2,s] - perf_dark_list[:,2,s] + perf_list[:,3,s] - perf_dark_list[:,3,s]) 
    mutka = (perf_list[:,0,c] - perf_dark_list[:,0,c] +  perf_list[:,1,c]  - perf_dark_list[:,1,c] + perf_list[:,2,c]  - perf_dark_list[:,2,c]  + perf_list[:,3,c]  - perf_dark_list[:,3,c]) 
    print "suora vs mutk norm:", stats.wilcoxon(suora, mutka)


    
    straight = np.array([corr[:,0], corr_d[:,0]])
    straight = straight.T

    cornering = np.array([corr[:,1], corr_d[:,1]])
    cornering = cornering.T

    plt.figure()
    
    plt.plot(straight[0], 'o--', label = "ylin suun")
    plt.plot(straight[1], 'o--', label = "ylin ~suun")
    plt.plot(straight[2], 'o--', label = "alin suun")
    plt.plot(straight[3], 'o--', label = "alin ~suun")

    plt.plot(x2, cornering[0], 'o--', color = "blue")
    plt.plot(x2, cornering[1], 'o--', color = "green")
    plt.plot(x2, cornering[2], 'o--', color = "red")
    plt.plot(x2, cornering[3], 'o--', color = "cyan")


    x_labs2 = ["straight","straight_control", "cornering", "cornering_control"]
    x_labs2 = ["suora","mutka", "suora ()", "mutka ()"]

    plt.ylabel("suoritus %")
    plt.xticks(x3, x_labs2)
    plt.xlim(-0.5, 4.5)
    plt.legend()

    a0 = [np.mean(perf_list[:,0,c]), np.std(perf_list[:,0,c], ddof = 1), np.mean(perf_dark_list[:,0,c]), np.std(perf_dark_list[:,0,c], ddof = 1)]
    a0 = [round(elem, 3) for elem in a0]
    b0 = [np.mean(perf_list[:,1,c]), np.std(perf_list[:,1,c], ddof = 1), np.mean(perf_dark_list[:,1,c]), np.std(perf_dark_list[:,1,c], ddof = 1)]
    b0 = [round(elem, 3) for elem in b0]
    c0 = [np.mean(perf_list[:,2,c]), np.std(perf_list[:,2,c], ddof = 1), np.mean(perf_dark_list[:,2,c]), np.std(perf_dark_list[:,2,c], ddof = 1)]
    c0 = [round(elem, 3) for elem in c0]
    d0 = [np.mean(perf_list[:,3,c]), np.std(perf_list[:,3,c], ddof = 1), np.mean(perf_dark_list[:,3,c]), np.std(perf_dark_list[:,3,c], ddof = 1)]
    d0 = [round(elem, 3) for elem in d0]

    

    df = DataFrame({'Ylos suun': a0, 'Ylos ~suun': b0, 'Alas suun': c0, 'Alas ~suun': d0})
    df.to_excel('taulukko.xlsx', sheet_name='sheet1', index=False)

    df = DataFrame({'Ylos suun': perf_dark_list[:,0,s], 'Ylos ~suun': perf_dark_list[:,1,s], 'Alas suun': perf_dark_list[:,2,s], 'Alas ~suun': perf_dark_list[:,3,s]})
    df.to_excel('DarkSuora.xlsx', sheet_name='sheet1', index=False)


    df = DataFrame({'Ylos suun': perf_dark_list[:,0,c], 'Ylos ~suun': perf_dark_list[:,1,c], 'Alas suun': perf_dark_list[:,2,c], 'Alas ~suun': perf_dark_list[:,3,c]})
    df.to_excel('DarkMutka.xlsx', sheet_name='sheet1', index=False)

    df = DataFrame({'Ylos suun': perf_list[:,0,s], 'Ylos ~suun': perf_list[:,1,s], 'Alas suun': perf_list[:,2,s], 'Alas ~suun': perf_list[:,3,s]})
    df.to_excel('Suora.xlsx', sheet_name='sheet1', index=False)


    df = DataFrame({'Ylos suun': perf_list[:,0,c], 'Ylos ~suun': perf_list[:,1,c], 'Alas suun': perf_list[:,2,c], 'Alas ~suun': perf_list[:,3,c]})
    df.to_excel('Mutka.xlsx', sheet_name='sheet1', index=False)

    perf_list1 = perf_list - perf_dark_list

    a0 = [np.mean(perf_list1[:,0,s]), np.std(perf_list1[:,0,s], ddof = 1), np.mean(perf_list1[:,0,c]), np.std(perf_list1[:,0,c], ddof = 1)]
    a0 = [round(elem, 3) for elem in a0]
    b0 = [np.mean(perf_list1[:,1,s]), np.std(perf_list1[:,1,s], ddof = 1), np.mean(perf_list1[:,1,c]), np.std(perf_list1[:,1,c], ddof = 1)]
    b0 = [round(elem, 3) for elem in b0]
    c0 = [np.mean(perf_list1[:,2,s]), np.std(perf_list1[:,2,s], ddof = 1), np.mean(perf_list1[:,2,c]), np.std(perf_list1[:,2,c], ddof = 1)]
    c0 = [round(elem, 3) for elem in c0]
    d0 = [np.mean(perf_list1[:,3,s]), np.std(perf_list1[:,3,s], ddof = 1), np.mean(perf_list1[:,3,c]), np.std(perf_list1[:,3,c], ddof = 1)]
    d0 = [round(elem, 3) for elem in d0]

    df = DataFrame({'Ylos suun': a0, 'Ylos ~suun': b0, 'Alas suun': c0, 'Alas ~suun': d0})
    df.to_excel('taulukko2.xlsx', sheet_name='sheet1', index=False)

    #plt.rc('font', size=18)
    #plt.rc('xtick', labelsize=16)   
    #plt.rc('ytick', labelsize=16)
    #plt.rc('legend', fontsize=16)

    #plt.figure()
    #plt.hist(react_list[:,0,s], bins = 5, alpha = 0.5)
    #plt.figure()
    #plt.hist(react_dark_list[:,0,s], bins = 5, alpha = 0.5)

    #print stats.ttest_rel(perf_list[:,0,s], perf_dark_list[:,0,s])
    #print stats.ttest_rel(perf_list[:,1,s], perf_dark_list[:,1,s])
    #print stats.ttest_rel(perf_list[:,2,s], perf_dark_list[:,2,s])
    #print stats.ttest_rel(perf_list[:,3,s], perf_dark_list[:,3,s])

    #print stats.ttest_rel(perf_list[:,0,c], perf_dark_list[:,0,c])
    #print stats.ttest_rel(perf_list[:,1,c], perf_dark_list[:,1,c])
    #print stats.ttest_rel(perf_list[:,2,c], perf_dark_list[:,2,c])
    #print stats.ttest_rel(perf_list[:,3,c], perf_dark_list[:,3,c])

    #print stats.ttest_rel(react_list[:,0,s], react_dark_list[:,0,s])
    #print stats.ttest_rel(react_list[:,1,s], react_dark_list[:,1,s])
    #print stats.ttest_rel(react_list[:,2,s], react_dark_list[:,2,s])
    #print stats.ttest_rel(react_list[:,3,s], react_dark_list[:,3,s])

    #print stats.ttest_rel(react_list[:,0,c], react_dark_list[:,0,c])
    #print stats.ttest_rel(react_list[:,1,c], react_dark_list[:,1,c])
    #print stats.ttest_rel(react_list[:,2,c], react_dark_list[:,2,c])
    #print stats.ttest_rel(react_list[:,3,c], react_dark_list[:,3,c])
    
    fig = plt.figure("ajo")

    adjustFigAspect(fig, aspect = 0.9)
    #plt.rc('axes', titlesize=18) 
    #plt.rc('axes', labelsize=18)
    #plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=18)    # legend fontsize

    plt.plot([0.2,1.2], corr[0], 'o--', label = "S1/M1", color = "darkred", markersize = 10)
    plt.errorbar([0.2,1.2], corr[0], fmt='--', yerr=[0.5*np.std(perf_list[:,0,s], ddof = 1), 0.5*np.std(perf_list[:,0,c], ddof = 1)], color = "darkred", markersize = 10)
    plt.plot(corr[1], '*--', label = "S2/M2", color = "tomato", markersize = 10)
    plt.errorbar([0,1], corr[1], fmt='--', yerr=[0.5*np.std(perf_list[:,1,s], ddof = 1), 0.5*np.std(perf_list[:,1,c], ddof = 1)], color = "tomato", markersize = 10)
    plt.plot([0.3,1.3], corr[2], 'p--', label = "S3/M3", color = "darkolivegreen", markersize = 10)
    plt.errorbar([0.3,1.3], corr[2], fmt='--', yerr=[0.5*np.std(perf_list[:,2,s], ddof = 1), 0.5*np.std(perf_list[:,2,c], ddof = 1)], color = "darkolivegreen")
    plt.plot([0.1,1.1], corr[3], 's--', label = "S4/M4", color = "greenyellow", markersize = 10)
    plt.errorbar([0.1,1.1], corr[3], fmt='--', yerr=[0.5*np.std(perf_list[:,3,s], ddof = 1), 0.5*np.std(perf_list[:,3,c], ddof = 1)], color = "greenyellow")

    np.save("ajo.npy",corr)

    plt.plot([2.2,3.2], corr_d[0], 'o--', color = "darkred", markersize = 10)
    plt.errorbar([2.2,3.2], corr_d[0], fmt='--', yerr=[0.5*np.std(perf_dark_list[:,0,s], ddof = 1), 0.5*np.std(perf_dark_list[:,0,c], ddof = 1)], color = "darkred")

    plt.plot(x2, corr_d[1], '*--', color = "tomato", markersize = 10)
    plt.errorbar(x2, corr_d[1], fmt='--', yerr=[0.5*np.std(perf_dark_list[:,1,s], ddof = 1), 0.5*np.std(perf_dark_list[:,1,c], ddof = 1)], color = "tomato")

    plt.plot([2.3,3.3], corr_d[2], 'p--', color = "darkolivegreen", markersize = 10)
    plt.errorbar([2.3,3.3], corr_d[2], fmt='--', yerr=[0.5*np.std(perf_dark_list[:,2,s], ddof = 1), 0.5*np.std(perf_dark_list[:,2,c], ddof = 1)], color = "darkolivegreen")
    plt.plot([2.1,3.1], corr_d[3], 's--', color = "greenyellow", markersize = 10)
    plt.errorbar([2.1,3.1], corr_d[3], fmt='--', yerr=[0.5*np.std(perf_dark_list[:,3,s], ddof = 1), 0.5*np.std(perf_dark_list[:,3,c], ddof = 1)], color = "greenyellow")

    np.save("dark.npy",corr_d)

    plt.ylabel("suoritustarkkuus (%)")
    plt.xticks(x3, x_labs)
    plt.xlim(-0.5, 4.0)
    plt.savefig('suoritus.png', dpi = 300, bbox_inches='tight', pad_inches = 0)

    plt.legend()



    react = np.delete(react, [1,3], 1)
    react_dark = np.delete(react_dark, [1,3], 1)
    fig = plt.figure("ajo_reaktio")
    adjustFigAspect(fig, aspect = 0.9)

    plt.plot([0.2,1.2], react[0], 'o--', label = "S1/M1", markersize = 10,  color = "darkred",)
    plt.errorbar([0.2,1.2], react[0], fmt='--', yerr=[0.5*np.std(react_list[:,0,s], ddof = 1), 0.5*np.std(react_list[:,0,c], ddof = 1)], color = "darkred")
    plt.plot(react[1], '*--', label = "S2/M2", color = "tomato", markersize = 10)
    plt.errorbar([0,1], react[1], fmt='--', yerr=[0.5*np.std(react_list[:,1,s], ddof = 1), 0.5*np.std(react_list[:,1,c], ddof = 1)], color = "tomato")
    plt.plot([0.3,1.3], react[2], 'p--', label = "S3/M3", color = "darkolivegreen", markersize = 10)
    plt.errorbar([0.3,1.3], react[2], fmt='--', yerr=[0.5*np.std(react_list[:,2,s], ddof = 1), 0.5*np.std(react_list[:,2,c], ddof = 1)], color = "darkolivegreen")
    plt.plot([0.1,1.1], react[3], 's--', label = "S4/M4", color = "greenyellow", markersize = 10)
    plt.errorbar([0.1,1.1], react[3], fmt='--', yerr=[0.5*np.std(react_list[:,3,s], ddof = 1), 0.5*np.std(react_list[:,3,c], ddof = 1)], color = "greenyellow")
    #np.save("ajo.npy",react)

    plt.plot([2.2,3.2], react_dark[0], 'o--',  color = "darkred", markersize = 10)
    plt.errorbar([2.2,3.2], react_dark[0], fmt='--', yerr=[0.5*np.std(react_dark_list[:,0,s], ddof = 1), 0.5*np.std(react_dark_list[:,0,c], ddof = 1)], color = "darkred")
    plt.plot(x2, react_dark[1], '*--', color = "tomato", markersize = 10)
    plt.errorbar(x2, react_dark[1], fmt='--', yerr=[0.5*np.std(react_dark_list[:,1,s], ddof = 1), 0.5*np.std(react_dark_list[:,1,c], ddof = 1)], color = "tomato")
    plt.plot([2.3,3.3], react_dark[2], 'p--', color = "darkolivegreen", markersize = 10)
    plt.errorbar([2.3,3.3], react_dark[2], fmt='--', yerr=[0.5*np.std(react_dark_list[:,2,s], ddof = 1), 0.5*np.std(react_dark_list[:,2,c], ddof = 1)], color = "darkolivegreen")
    plt.plot([2.1,3.1], react_dark[3], 's--', color = "greenyellow", markersize = 10)
    plt.errorbar([2.1,3.1], react_dark[3], fmt='--', yerr=[0.5*np.std(react_dark_list[:,3,s], ddof = 1), 0.5*np.std(react_dark_list[:,3,c], ddof = 1)], color = "greenyellow")


    #np.save("dark.npy",react_dark)


    plt.ylabel("reaktio (s)")
    plt.xticks(x3, x_labs)
    plt.xlim(-0.5, 4.0)

    plt.legend()

    plt.savefig('reaktio.png', dpi = 300, bbox_inches='tight', pad_inches = 0)

    #plt.show()
    

    plt.figure()
    fig, axes = plt.subplots( 2)
    


    #figure = plt.figure("gaze norm")
    new_data.append([-960,-540,0,0])
    new_data.append([960,540,0,0])
    data = np.array(new_data)
    data = data[data[:,3] == 0]
    #data = new_data[(new_data[:,0] > -1000) & (new_data[:,1] > -1000) ]
    #data = data[(data[:,0] < 1000.) & (data[:,1] < 1000.)]
    #print data[:,0]

    x = (data[:,0] + 960).astype(int)
    y = (data[:,1] + 540).astype(int)
    condition = (x <= 1920) & (y <= 1080) & (x >= 0) & (y >= 0)

    x = x[condition]
    y = y[condition]

    r = (7.5 / 70.0) * 1920

    xr = r * math.cos(math.pi/4.0)
    yr = r * math.sin(math.pi/4.0)
        
    #fig = plt.gcf()
    fix_r = 2.75/70. * 1920
    axes[0].add_patch(patches.Circle((960,540),fix_r, color='black', fill=False))

    #fig.gca().add_patch(patches.Circle((960 + r,540),55, color='black', fill=False))
    #fig.gca().add_patch(patches.Circle((960 - r,540),55, color='black', fill=False))

    #fig.gca().add_patch(patches.Circle((960 + xr,540 - yr),55, color='black', fill=False))
    #fig.gca().add_patch(patches.Circle((960 - xr,540 - yr),55, color='black', fill=False))


    def lines(x,y):
        r = (2.5 / 70.0) * 1920
        l = (r**2*2)**0.5 / 2.
        return np.array([(x - l, y),(x, y + l),(x + l, y),(x, y - l), (x - l, y)])
    sq1 = lines(960 + r,540)
    sq2 = lines(960 - r,540)
    sq3 = lines(960 + xr,540 - yr)
    sq4 = lines(960 - xr,540 - yr)

    axes[0].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "S1/M1")
    axes[0].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "S2/M2")
    axes[0].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "S3/M3")
    axes[0].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "S4/M4")

    #plt.scatter(x,y)
    h = np.max(x) - np.min(x)
    w = np.max(y) - np.min(y)
    #bins = max(min(int((h*w) / 1000.), 4000), 100)
    #
    H, xedges, yedges, img0 = axes[0].hist2d(x, y, bins = 400)
    axes[0].set_ylim([0, 1080])
    axes[0].set_xlim([0, 1920])
    #axes[0].scatter(x,y, color = 'purple', s = 5, alpha = 0.3)
    axes[0].set_xlabel(u'x-koordinaatti (pikseleinä)')
    axes[0].set_ylabel(u'y-koordinaatti (pikseleinä)')
    axes[0].legend()
    #plt.axes().set_aspect('equal')
    #plt.savefig('foo1.png', dpi = 300, bbox_inches='tight', pad_inches = 0)



    #figure = plt.figure("gaze control")
    new_data.append([-960,-540,0,1])
    new_data.append([960,540,0,1])
    data = np.array(new_data)
    data = data[data[:,3] == 1]
    #data = new_data[(new_data[:,0] > -1000) & (new_data[:,1] > -1000) ]
    #data = data[(data[:,0] < 1000.) & (data[:,1] < 1000.)]
    #print data[:,0]

    x = (data[:,0] + 960).astype(int)
    y = (data[:,1] + 540).astype(int)
    condition = (x <= 1920) & (y <= 1080) & (x >= 0) & (y >= 0)

    x = x[condition]
    y = y[condition]
    print len(x)

    r = (7.5 / 70.0) * 1920

    xr = r * math.cos(math.pi/4.0)
    yr = r * math.sin(math.pi/4.0)
        
    #fig = plt.gcf()
    axes[1].add_patch(patches.Circle((960,540),fix_r, color='black', fill=False))

    #fig.gca().add_patch(patches.Circle((960 + r,540),55, color='black', fill=False))
    #fig.gca().add_patch(patches.Circle((960 - r,540),55, color='black', fill=False))
    #fig.gca().add_patch(patches.Circle((960 + xr,540 - yr),55, color='black', fill=False))
    #fig.gca().add_patch(patches.Circle((960 - xr,540 - yr),55, color='black', fill=False))



    axes[1].plot(sq1[:,0], sq1[:,1], linewidth = 2, color = "darkred", label = "S1/M1")
    axes[1].plot(sq2[:,0], sq2[:,1], linewidth = 2, color = "tomato", label = "S2/M2")
    axes[1].plot(sq3[:,0], sq3[:,1], linewidth = 2, color = "darkolivegreen", label = "S2/M2")
    axes[1].plot(sq4[:,0], sq4[:,1], linewidth = 2, color = "greenyellow", label = "S3/M3")


    #plt.scatter(x,y)
    h = np.max(x) - np.min(x)
    w = np.max(y) - np.min(y)
    #bins = max(min(int((h*w) / 1000.), 4000), 100)
    #
    from matplotlib.colors import LogNorm
    axes[1].set_ylim(0, 1080)
    axes[1].set_xlim(0, 1920)
    H, xedges, yedges, img1 = axes[1].hist2d(x, y, bins = 400)
    
    #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #im = axes[1].imshow(H, cmap=plt.cm.jet, extent=extent, norm=LogNorm())
    fig.colorbar(img1, ax=axes[1], shrink=0.9, pad = 0.01)
    fig.colorbar(img0, ax=axes[0], shrink=0.9, pad = 0.01)

    #axes[1].scatter(x,y, color = 'purple', s = 5, alpha = 0.3)
    #plt.xlabel('x')
    #plt.ylabel('y')
    axes[1].set_xlabel(u'x-koordinaatti (pikseleinä)')
    axes[1].set_ylabel(u'y-koordinaatti (pikseleinä)')
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    #plt.subplots_adjust(hspace = 0)
    axes[0].title.set_text(u'Ohjaustehtävä')
    axes[1].title.set_text(u'Verrokkitehtävä')

    #plt.savefig('foo2.png', dpi = 300, bbox_inches='tight', pad_inches = 0)

    plt.show()

            
    
