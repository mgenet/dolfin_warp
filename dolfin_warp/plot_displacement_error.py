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

def plot_displacement_error(
        working_folder,
        working_basenames,
        suffix="",
        verbose=1):

    n_frames = len(open(working_folder+"/"+working_basenames[0]+"-error.dat").readlines())-1
    #print("n_frames = " + str(n_frames))

    plotfile = open("plot_displacement_error"+("-"+suffix)*(suffix!="")+".plt", "w")
    plotfile.write('''\
set terminal pdf enhanced size 5,3

set output "plot_displacement_error'''+('''-'''+suffix)*(suffix!="")+'''.pdf"

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

    plotfile.write('''\
set xlabel "frame number"
set xrange [0:'''+str(n_frames-1)+''']

set ylabel "displacement error (%)"
set yrange [0:*]

''')

    for k_basename in range(len(working_basenames)):
        working_basename = working_basenames[k_basename]
        plotfile.write('''\
'''+(k_basename==0)*('''plot ''')+(k_basename>0)*('''     ''')+'''"'''+working_folder+'''/'''+working_basename+'''-error.dat" using ($1):(100*$4) with lines linestyle '''+str(k_basename+1)+''' linewidth 5 title "'''+working_basename+'''"'''+(k_basename<len(working_basenames)-1)*(''',\\
''')+(k_basename==len(working_basenames)-1)*('''

'''))

    plotfile.close()

    os.system("gnuplot plot_displacement_error"+("-"+suffix)*(suffix!="")+".plt")
