#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import os

########################################################################

def plot_strains(
        working_folder,
        working_basenames,
        working_scalings=None,
        ref_folder=None,
        ref_basename=None,
        components="all", # all, circ-long, or rad-circ
        suffix="",
        verbose=1):

    if (working_scalings is not None):
        assert (len(working_scalings) == len(working_basenames))
    else:
        working_scalings = [1.] * len(working_basenames)

    if (ref_folder is not None) and (ref_basename is not None):
        lines = open(ref_folder+"/"+ref_basename+"-strains.dat").readlines()
    else:
        lines = open(working_folder+"/"+working_basenames[0]+"-strains.dat").readlines()
    n_frames = len(lines)-1
    n_sectors = (len(lines[-1].split(" "))-1)/12
    #print "n_frames = " + str(n_frames)
    #print "n_sectors = " + str(n_sectors)

    assert (components in ("all", "circ-long", "rad-circ"))
    if (components == "all"):
        comp_names = ["radial", "circumferential", "longitudinal", "radial-circumferential", "radial-longitudinal", "circumferential-longitudinal"]
    elif (components == "circ-long"):
        comp_names = ["circumferential","longitudinal","circumferential-longitudinal"]
    elif (components == "rad-circ"):
        comp_names = ["radial", "circumferential", "radial-circumferential"]

    if (suffix is None):
        plotfile_basename = working_folder+"/"+working_basenames[0]+"-strains"
    else:
        plotfile_basename = "plot_strains"+("-"+suffix)*(suffix!="")
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

set key box textcolor variable width +1

set grid

''')

    for k_sector in range(n_sectors):
        plotfile.write('''\
set multiplot layout '''+('''2''')*(components=="all")+('''1''')*(components in ("circ-long", "rad-circ"))+''',3

set xlabel "frame ()"
set xrange [0:'''+str(n_frames)+''']

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
''')
            if ("-" in comp_name):
                plotfile.write('''\
set yrange [-10:10]
''')
            else:
                plotfile.write('''\
set yrange [-30:30]

''')
            plotfile.write('''\
plot 0 linecolor rgb "black" notitle,\\
''')
            if (ref_folder is not None) and (ref_basename is not None):
                plotfile.write('''\
     "'''+ref_folder+'''/'''+ref_basename+'''-strains.dat" using ($1):(100*$'''+str(2+12*k_sector+2*k_comp)+'''):(100*$'''+str(2+12*k_sector+2*k_comp+1)+''') with lines linecolor "black" linewidth 5 notitle,\
     "'''+ref_folder+'''/'''+ref_basename+'''-strains.dat" using ($1):(100*$'''+str(2+12*k_sector+2*k_comp)+'''):(100*$'''+str(2+12*k_sector+2*k_comp+1)+''') with errorbars linecolor "black" pointtype 1 linewidth 1 notitle'''+(len(working_basenames)>0)*(''',\\
''')+(len(working_basenames)==0)*('''

'''))
            for k_basename in range(len(working_basenames)):
                working_basename = working_basenames[k_basename]
                working_scaling = working_scalings[k_basename]
                plotfile.write('''\
     "'''+working_folder+'''/'''+working_basename+'''-strains.dat" using ('''+str(working_scaling)+'''*($1)+'''+str(k_basename)+'''./10):(100*$'''+str(2+12*k_sector+2*k_comp)+'''):(100*$'''+str(2+12*k_sector+2*k_comp+1)+''') with lines linestyle '''+str(k_basename+1)+''' linewidth 5 title "'''+working_basename+'''",\
     "'''+working_folder+'''/'''+working_basename+'''-strains.dat" using ('''+str(working_scaling)+'''*($1)+'''+str(k_basename)+'''./10):(100*$'''+str(2+12*k_sector+2*k_comp)+'''):(100*$'''+str(2+12*k_sector+2*k_comp+1)+''') with errorbars linestyle '''+str(k_basename+1)+''' linewidth 1 notitle'''+(k_basename<len(working_basenames)-1)*(''',\\
''')+(k_basename==len(working_basenames)-1)*('''

'''))

    plotfile.close()

    os.system("gnuplot "+plotfile_basename+".plt")
