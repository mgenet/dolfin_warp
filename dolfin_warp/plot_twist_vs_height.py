#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import os

################################################################################

def plot_twist_vs_height(
        working_folder,
        working_basename,
        beta_range=15.,
        suffix="",
        verbose=1):

    lines = open(working_folder+"/"+working_basename+"-strains.dat").readlines()[1:]
    n_frames = len(lines)

    if (suffix is None):
        plotfile_basename = working_folder+"/"+working_basename+"-twist_vs_height"
    else:
        plotfile_basename = "plot_twist_vs_height"+("-"+suffix)*(suffix!="")
    plotfile = open(plotfile_basename+".plt", "w")

    plotfile.write('''\
set terminal pdf enhanced

load "Set1.plt"

set output "'''+plotfile_basename+'''.pdf"

set key off
# set key box textcolor variable width +0

set grid

set xlabel "beta (deg)"
xrange = '''+str(beta_range)+'''
set xrange [-xrange:+xrange]

# set ylabel "ll ()"
set yrange [0.:1.]
set ytics("apex" 0, "base" 1)

# f(x) = a*x + b

# g(x) = x/c - d/c
# h(x) = c*x + d

g(x) = c*x + d
h(x) = x/c - d/c

FIT_MAXITER = 10

datafile = "'''+working_folder+'''/'''+working_basename+'''-twist_vs_height.dat"

''')
    for k_frame in range(n_frames):
        plotfile.write('''\
# a = 1.
# b = 0.5

# c = 1.
# d = 0.5

c = 1.
d = -0.5*c

set title "index '''+str(k_frame)+'''"
# fit f(x) datafile using 3:2 index '''+str(k_frame)+''' via a,b
'''+('''# ''')*(k_frame==0)+'''fit g(x) datafile using 2:3 index '''+str(k_frame)+''' via c,d
plot datafile using 3:2 index '''+str(k_frame)+''' notitle, h(x) notitle

''')

    plotfile.close()

    os.system("gnuplot "+plotfile_basename+".plt")
