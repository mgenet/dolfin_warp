#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import os

################################################################################

def plot_strains_vs_radius(
        working_folder,
        working_basenames,
        ref_folder=None,
        ref_basename=None,
        components="all", # all, circ-long, or rad-circ
        yranges=[0]*6,
        suffix="",
        verbose=1):

    if (ref_folder is not None) and (ref_basename is not None):
        lines = open(ref_folder+"/"+ref_basename+"-strains.dat").readlines()[1:]
    else:
        lines = open(working_folder+"/"+working_basenames[0]+"-strains.dat").readlines()[1:]
    n_frames = len(lines)

    comp_names = ["radial", "circumferential", "longitudinal", "radial-circumferential", "radial-longitudinal", "circumferential-longitudinal"]
    assert (components in ("all", "circ-long", "rad-circ"))
    if (components == "all"):
        comp_indx = [0,1,2,3,4,5]
        n_cols = 3
        n_rows = 2
    elif (components == "circ-long"):
        comp_indx = [1,2,5]
        n_cols = 3
        n_rows = 1
    elif (components == "rad-circ"):
        comp_indx = [0,1,3]
        n_cols = 3
        n_rows = 1

    if (suffix is None):
        plotfile_basename = working_folder+"/"+working_basenames[0]+"-strains_vs_radius"
    else:
        plotfile_basename = "plot_strains_vs_radius"+("-"+suffix)*(suffix!="")
    plotfile = open(plotfile_basename+".plt", "w")

    size_x = 5
    size_y = 3
    plotfile.write('''\
set terminal pdf enhanced size '''+str(n_cols*size_x)+''','''+str(n_rows*size_y)+'''

set output "'''+plotfile_basename+'''.pdf"

load "Set1.plt"

set key box textcolor variable width +0

set grid

''')
    for k_frame in range(n_frames):
        plotfile.write('''\
set multiplot layout '''+str(n_rows)+''','''+str(n_cols)+'''

set xrange [0.:1.]
set xtics("endocardium" 0, "epicardium" 1)

''')
        for k_comp in comp_indx:
            plotfile.write('''\
set ylabel "'''+comp_names[k_comp]+''' strain (%)"
''')
            yrange = yranges[k_comp]
            if (yrange > 0):
                plotfile.write('''\
set yrange [-'''+str(yrange)+''':'''+str(yrange)+''']

plot 0 linecolor rgb "black" notitle,\\
''')
            if (ref_folder is not None) and (ref_basename is not None):
                plotfile.write('''\
     "'''+ref_folder+'''/'''+ref_basename+'''-strains_vs_radius.dat" using ($2):(100*$'''+str(3+k_comp)+''') index '''+str(k_frame)+''' with points linecolor "black" linewidth 3 pointtype 1 pointsize 1 notitle'''+(len(working_basenames)>0)*(''',\\
''')+(len(working_basenames)==0)*('''

'''))
            for k_basename in range(len(working_basenames)):
                working_basename = working_basenames[k_basename]
                plotfile.write('''\
     "'''+working_folder+'''/'''+working_basename+'''-strains_vs_radius.dat" using ($2):(100*$'''+str(3+k_comp)+''') index '''+str(k_frame)+''' with points linestyle '''+str(k_basename+1)+''' linewidth 3 pointtype 1 pointsize 1 title "'''+working_basename+'''"'''+(k_basename<len(working_basenames)-1)*(''',\\
''')+(k_basename==len(working_basenames)-1)*('''

'''))

    plotfile.close()

    os.system("gnuplot "+plotfile_basename+".plt")
