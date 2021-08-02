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
import sys

################################################################################

def plot_binned_strains_vs_radius(
        working_folder,
        working_basenames,
        ref_folder=None,
        ref_basename=None,
        components="all",
        suffix="",
        verbose=1):

    if (ref_folder is not None) and (ref_basename is not None):
        lines = open(ref_folder+"/"+ref_basename+"-strains.dat").readlines()[1:]
    else:
        lines = open(working_folder+"/"+working_basenames[0]+"-strains.dat").readlines()[1:]
    n_frames = len(lines)

    assert (components in ("all", "circ-long", "rad-circ"))
    if (components == "all"):
        comp_names = ["radial", "circumferential", "longitudinal", "radial-circumferential", "radial-longitudinal", "circumferential-longitudinal"]
    elif (components == "circ-long"):
        comp_names = ["circumferential","longitudinal","circumferential-longitudinal"]
    elif (components == "rad-circ"):
        comp_names = ["radial", "circumferential", "radial-circumferential"]

    if (suffix is None):
        plotfile_basename = working_folder+"/"+working_basenames[0]+"-binned_strains_vs_radius"
    else:
        plotfile_basename = "plot_binned_strains_vs_radius"+("-"+suffix)*(suffix!="")

    plotfile = open(plotfile_basename+".plt", "w")
    plotfile.write('''\
set terminal pdf enhanced size 15,'''+('''6''')*(components=="all")+('''3''')*(components in ("circ-long", "rad-circ"))+'''

set output "'''+plotfile_basename+'''.pdf"

load "Set1.plt"
set linestyle 1 pointtype 1
set linestyle 2 pointtype 1
set linestyle 3 pointtype 1
set linestyle 4 pointtype 1
set linestyle 5 pointtype 1
set linestyle 6 pointtype 1
set linestyle 7 pointtype 1
set linestyle 8 pointtype 1
set linestyle 9 pointtype 1

set key box textcolor variable width +0

set grid

''')
    for k_frame in range(n_frames):
        plotfile.write('''\
set multiplot layout '''+('''2''')*(components=="all")+('''1''')*(components in ("circ-long", "rad-circ"))+''',3

set xrange [0.:1.]
set xtics("endocardium" 0, "epicardium" 1)

''')
        for comp_name in comp_names:
            if   (comp_name == "radial"                      ): k_comp = 0
            elif (comp_name == "circumferential"             ): k_comp = 1
            elif (comp_name == "longitudinal"                ): k_comp = 2
            elif (comp_name == "radial-circumferential"      ): k_comp = 3
            elif (comp_name == "radial-longitudinal"         ): k_comp = 4
            elif (comp_name == "circumferential-longitudinal"): k_comp = 5
            plotfile.write('''\
set ylabel "'''+comp_name+''' strain (%)"
set yrange [-50:50]

plot 0 linecolor rgb "black" notitle,\\
''')
            if (ref_folder is not None) and (ref_basename is not None):
                plotfile.write('''\
     "'''+ref_folder+'''/'''+ref_basename+'''-binned_strains_vs_radius.dat" using ($2):(100*$'''+str(3+2*k_comp)+''') index '''+str(k_frame)+''' with lines linecolor "black" linewidth 3 notitle,\\
     "'''+ref_folder+'''/'''+ref_basename+'''-binned_strains_vs_radius.dat" using ($2):(100*$'''+str(3+2*k_comp)+'''):(100*$'''+str(3+2*k_comp+1)+''') index '''+str(k_frame)+''' with errorbars linecolor "black" linewidth 1 notitle'''+(len(working_basenames)>0)*(''',\\
''')+(len(working_basenames)==0)*('''

'''))
            for k_basename in range(len(working_basenames)):
                working_basename = working_basenames[k_basename]
                plotfile.write('''\
     "'''+working_folder+'''/'''+working_basename+'''-binned_strains_vs_radius.dat" using ($2):(100*$'''+str(3+2*k_comp)+''') index '''+str(k_frame)+''' with lines linestyle '''+str(k_basename+1)+''' linewidth 3 title "'''+working_basename+'''",\\
     "'''+working_folder+'''/'''+working_basename+'''-binned_strains_vs_radius.dat" using ($2):(100*$'''+str(3+2*k_comp)+'''):(100*$'''+str(3+2*k_comp+1)+''') index '''+str(k_frame)+''' with errorbars linestyle '''+str(k_basename+1)+''' linewidth 1 notitle'''+(k_basename<len(working_basenames)-1)*(''',\\
''')+(k_basename==len(working_basenames)-1)*('''

'''))

    plotfile.close()

    os.system("gnuplot "+plotfile_basename+".plt")
