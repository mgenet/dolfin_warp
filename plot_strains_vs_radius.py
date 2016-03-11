#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import math
import numpy
import os
import sys

########################################################################

def plot_strains_vs_radius(
        working_folder,
        working_basenames,
        index=10,
        ref_folder=None,
        ref_basename=None,
        components="all",
        suffix="",
        verbose=1):

    assert (components in ("all", "circ-long", "rad-circ"))

    plotfile = open("plot_strains_vs_radius"+("-"+suffix)*(suffix!="")+".plt", "w")
    plotfile.write('''\
set terminal pdf enhanced size 15,'''+('''6''')*(components=="all")+('''3''')*(components in ("circ-long", "rad-circ"))+'''

set output "plot_strains_vs_radius'''+('''-'''+suffix)*(suffix!="")+'''.pdf"

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

set multiplot layout '''+('''2''')*(components=="all")+('''1''')*(components in ("circ-long", "rad-circ"))+''',3

''')
    if (components == "all"):
        comp_names = ["radial", "circumferential", "longitudinal", "radial-circumferential", "radial-longitudinal", "circumferential-longitudinal"]
    elif (components == "circ-long"):
        comp_names = ["circumferential","longitudinal","circumferential-longitudinal"]
    elif (components == "rad-circ"):
        comp_names = ["radial", "circumferential", "radial-circumferential"]
    else:
        assert (0), "components must be \"all\", \"circ-long\" or \"rad-circ\". Aborting."
    for comp_name in comp_names:
        if   (comp_name == "radial"                      ): k_comp = 0
        elif (comp_name == "circumferential"             ): k_comp = 1
        elif (comp_name == "longitudinal"                ): k_comp = 2
        elif (comp_name == "radial-circumferential"      ): k_comp = 3
        elif (comp_name == "radial-longitudinal"         ): k_comp = 4
        elif (comp_name == "circumferential-longitudinal"): k_comp = 5
        plotfile.write('''\

set xlabel "r (mm)"
set xrange [0.2:0.4]

set ylabel "'''+comp_name+''' strain (%)"
set yrange [-40:40]

plot 0 linecolor rgb "black" notitle,\\
''')
        if (ref_folder is not None) and (ref_basename is not None):
            plotfile.write('''\
     "'''+ref_folder+'''/'''+ref_basename+'''-strains_vs_radius.dat" using ($2):(100*$'''+str(3+k_comp)+''') index '''+str(index)+''' with points linecolor "black" pointtype 1 pointsize 1 linewidth 3 notitle'''+(len(working_basenames)>0)*(''',\\
''')+(len(working_basenames)==0)*('''

'''))
        for k_basename in range(len(working_basenames)):
            working_basename = working_basenames[k_basename]
            plotfile.write('''\
     "'''+working_folder+'''/'''+working_basename+'''-strains_vs_radius.dat" using ($2):(100*$'''+str(3+k_comp)+''') index '''+str(index)+''' with points linestyle '''+str(k_basename+1)+''' pointtype 1 pointsize 1 linewidth 3 title "'''+working_basename+'''"'''+(k_basename<len(working_basenames)-1)*(''',\\
''')+(k_basename==len(working_basenames)-1)*('''

'''))

    plotfile.close()

    os.system("gnuplot plot_strains_vs_radius"+("-"+suffix)*(suffix!="")+".plt")
