#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import math
import matplotlib.pyplot
import numpy

################################################################################

def plot_regional_strains(
        working_folder,
        working_basename,
        k_frame=None,
        components="all", # all, circ-long, or rad-circ
        yranges=[0]*6,
        n_sectors_c=6,
        n_sectors_l=3,
        suffix="",
        verbose=1):

    assert (components in ("all", "circ-long", "rad-circ"))
    if (components == "all"):
        comp_names = ["radial", "circumferential", "longitudinal", "radial-circumferential", "radial-longitudinal", "circumferential-longitudinal"]
        n_cols = 3
        n_rows = 2
    elif (components == "circ-long"):
        comp_names = ["circumferential","longitudinal","circumferential-longitudinal"]
        n_cols = 3
        n_rows = 1
    elif (components == "rad-circ"):
        comp_names = ["radial", "circumferential", "radial-circumferential"]
        n_cols = 3
        n_rows = 1
    n_comp = len(comp_names)

    strains_all = 100*numpy.loadtxt(working_folder+"/"+working_basename+"-strains.dat")[:,1:]

    n_frames = len(strains_all)
    if (k_frame is None):
        #k_frame = n_frames//2
        k_frame = numpy.argmin(strains_all[:,2])

    strains_es = strains_all[k_frame]
    #print("len(strains_es) = "+str(len(strains_es)))

    n_sectors = n_sectors_l * n_sectors_c
    assert (len(strains_es)//2 == (1+n_sectors)*n_comp), "Number of strain components ("+str(len(strains_es)//2)+") inconsistent with number of sectors (n_sectors_c="+str(n_sectors_c)+", n_sectors_l="+str(n_sectors_l)+"). Aborting."

    strains_es_avg = [[strains_es[(k_sector+1)*2*n_comp+2*k_comp] for k_sector in range(n_sectors)] for k_comp in range(n_comp)]

    size = 4
    fig = matplotlib.pyplot.figure(figsize=(n_cols*size,n_rows*size))

    for k_comp in range(n_comp):

        subplot = fig.add_subplot(n_rows, n_cols, k_comp+1, projection='polar')
        subplot.set_title(comp_names[k_comp]+" strain (%)")

        subplot.set_xticks([])
        subplot.set_yticks([])

        strains_comp = strains_es_avg[k_comp]

        yrange = yranges[k_comp]
        if (yrange == 0):
            strains_comp_min = min(strains_comp)
            strains_comp_max = max(strains_comp)
            strains_comp_min = min(strains_comp_min, -strains_comp_max)
            strains_comp_max = -strains_comp_min
        else:
            strains_comp_min = -yrange
            strains_comp_max = +yrange
        assert (strains_comp_max>strains_comp_min), "strains_comp_max ("+str(strains_comp_max)+") <= strains_comp_min (strains_comp_min). Aborting."

        cmap = matplotlib.pyplot.cm.get_cmap('coolwarm')
        smap = matplotlib.pyplot.cm.ScalarMappable(
            cmap=cmap,
            norm=matplotlib.pyplot.Normalize(vmin=strains_comp_min, vmax=strains_comp_max))
        smap._A = []

        cbar = matplotlib.pyplot.colorbar(
            mappable=smap,
            #orientation="horizontal",
            format="%+g",
            shrink=2./3)
        #cbar.set_label(comp_names[k_comp]+" (%)")
        cbar.solids.set_edgecolor("face")

        k_sector = 0
        for k_l in range(n_sectors_l):
            for k_c in range(n_sectors_c):
                subplot.bar(
                    left = k_c * 2*math.pi/n_sectors_c,
                    height = 1./n_sectors_l,
                    width = 2*math.pi/n_sectors_c,
                    bottom = 1.-float(k_l+1)/n_sectors_l,
                    color = cmap(float(strains_comp[k_sector]-strains_comp_min)/(strains_comp_max - strains_comp_min)),
                    linewidth=0)
                #subplot.annotate(
                    #"{:+2.1f}".format(strains_comp[k_sector]),
                    #xy=    [(k_c+0.5) * 2*math.pi/n_sectors_c, 1.-float(k_l+0.5)/n_sectors_l],
                    #xytext=[(k_c+0.5) * 2*math.pi/n_sectors_c, 1.-float(k_l+0.5)/n_sectors_l],
                    #xycoords='polar',
                    #textcoords='data',
                    #horizontalalignment='center',
                    #verticalalignment='center')
                subplot.annotate(
                    str(k_sector+1),
                    xy=    [(k_c+0.5) * 2*math.pi/n_sectors_c, 1.-float(k_l+0.5)/n_sectors_l],
                    xytext=[(k_c+0.5) * 2*math.pi/n_sectors_c, 1.-float(k_l+0.5)/n_sectors_l],
                    xycoords='polar',
                    textcoords='data',
                    horizontalalignment='center',
                    verticalalignment='center')
                k_sector += 1

        subplot.annotate(
            "anterior",
            textcoords='data',
            xycoords='polar',
            xy=    [1*math.pi/4,1.1],
            xytext=[1*math.pi/4,1.1],
            horizontalalignment='center',
            verticalalignment='center',
            rotation=-45)
        subplot.annotate(
            "septal",
            textcoords='data',
            xycoords='polar',
            xy=    [3*math.pi/4,1.1],
            xytext=[3*math.pi/4,1.1],
            horizontalalignment='center',
            verticalalignment='center',
            rotation=45)
        subplot.annotate(
            "inferior",
            textcoords='data',
            xycoords='polar',
            xy=    [5*math.pi/4,1.1],
            xytext=[5*math.pi/4,1.1],
            horizontalalignment='center',
            verticalalignment='center',
            rotation=-45)
        subplot.annotate(
            "lateral",
            textcoords='data',
            xycoords='polar',
            xy=    [7*math.pi/4,1.1],
            xytext=[7*math.pi/4,1.1],
            horizontalalignment='center',
            verticalalignment='center',
            rotation=45)

    matplotlib.pyplot.tight_layout()

    if (suffix is None):
        plotfile_basename = working_folder+"/"+working_basename+"-regional_strains"
    else:
        plotfile_basename = "plot_regional_strains"+("-"+suffix)*(suffix!="")
    #matplotlib.pyplot.savefig(plotfile_basename+".pdf", format='pdf')
    matplotlib.pyplot.savefig(plotfile_basename+".pdf", format='pdf', bbox_inches="tight")
