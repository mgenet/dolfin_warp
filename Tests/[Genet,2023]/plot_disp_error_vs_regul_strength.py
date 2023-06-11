#coding=utf8

########################################################################

import numpy
import os

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

########################################################################

def plot_disp_error_vs_regul_strength(
        images_folder                            : str          ,
        sol_folder                               : str          ,
        structure_type                           : str          ,
        deformation_type                         : str          ,
        texture_type                             : str          ,
        regul_type                               : str          ,
        noise_level_lst                          : list         ,
        n_runs_for_noisy_images                  : int          ,
        regul_level_lst                          : list         ,
        mesh_size                                : float = 0.1  ,
        regul_level_for_zero                     : float = 1e-3 ,
        generate_datafile                        : bool  = True ,
        generate_datafile_with_limited_precision : bool  = False,
        generate_plotfile                        : bool  = True ,
        generate_plot                            : bool  = True ):

    print ("images_folder:"   , images_folder   )
    print ("sol_folder:"      , sol_folder      )
    print ("structure_type:"  , structure_type  )
    print ("deformation_type:", deformation_type)
    print ("texture_type:"    , texture_type    )
    print ("regul_type:"      , regul_type      )

    script_basename  = "plot_disp_error_vs_regul_strength"
    if not os.path.exists(script_basename):
        os.mkdir(script_basename)
    
    datafile_basename  = script_basename
    datafile_basename += "/"+structure_type
    datafile_basename += "-"+deformation_type
    datafile_basename += "-"+texture_type
    datafile_basename += "-"+regul_type

########################################################################

    if (generate_datafile) or (generate_datafile_with_limited_precision):

        if (generate_datafile): data_printer = mypy.DataPrinter(
            names=["noise_level", "regul_level", "disp_err_avg", "disp_err_std"],
            filename=datafile_basename+".dat",
            limited_precision=False)

        if (generate_datafile_with_limited_precision): data_printer2 = mypy.DataPrinter(
            names=["noise_level", "regul_level", "disp_err_avg", "disp_err_std"],
            filename=datafile_basename+"-limited_precision.dat",
            limited_precision="True")

        if (generate_datafile): data_printer3 = mypy.DataPrinter(
            names=["noise_level", "regul_level", "disp_err"],
            filename=datafile_basename+"-all_points.dat",
            limited_precision=False)

        if (structure_type in ("square", "disc", "ring")):
            ref_disp_array_name = "displacement"
        elif (structure_type in ("heart")):
            ref_disp_array_name = "U"
        else: assert (0)

        for noise_level in noise_level_lst:
         for regul_level in regul_level_lst:

            print ("noise_level:", noise_level)
            print ("regul_level:", regul_level)

            if (regul_level == 0.0):
                regul_level_for_write = regul_level_for_zero
            else:
                regul_level_for_write = regul_level

            n_runs = n_runs_for_noisy_images if (noise_level > 0) else 1
            disp_err_lst = []

            for k_run in range(1, n_runs+1):

                print ("k_run:", k_run)

                images_basename  = structure_type
                images_basename += "-"+deformation_type
                images_basename += "-"+texture_type
                images_basename += "-noise="+str(noise_level)
                if (n_runs > 1):
                    images_basename += "-run="+str(k_run).zfill(2)

                sol_basename = images_basename
                sol_basename += "-h="+str(mesh_size)
                sol_basename += "-"+regul_type
                sol_basename += "-regul="+str(regul_level)

                disp_err = dwarp.compute_displacement_error(
                    working_folder=sol_folder,
                    working_basename=sol_basename,
                    ref_folder=images_folder,
                    ref_basename=structure_type+"-"+deformation_type+"-h=0.1",
                    working_disp_array_name="displacement",
                    ref_disp_array_name=ref_disp_array_name,
                    verbose=0)
                print ("disp_err:", disp_err)

                if (generate_datafile): data_printer3.write_line([noise_level, regul_level_for_write, disp_err])

                disp_err_lst.append(disp_err)

            disp_err_avg = numpy.mean(disp_err_lst)
            disp_err_std = numpy.std(disp_err_lst)
            print ("disp_err_avg:", disp_err_avg)
            print ("disp_err_std:", disp_err_std)

            if (generate_datafile                       ): data_printer.write_line([noise_level, regul_level_for_write, disp_err_avg, disp_err_std])
            if (generate_datafile_with_limited_precision): data_printer2.write_line([noise_level, regul_level_for_write, disp_err_avg, disp_err_std])

         if (generate_datafile                       ): data_printer.write_line(); data_printer.write_line()
         if (generate_datafile_with_limited_precision): data_printer2.write_line(); data_printer2.write_line()
         if (generate_datafile                       ): data_printer3.write_line(); data_printer3.write_line()

        if (generate_datafile                       ): data_printer.close()
        if (generate_datafile_with_limited_precision): data_printer2.close()
        if (generate_datafile                       ): data_printer3.close()

########################################################################

    if (generate_plotfile):

        plotfile = open(datafile_basename+".plt", "w")
        plotfile.write('''\
set terminal pdf enhanced size 4,3; datafile_ext = "pdf"

load "Set1.plt"

set linestyle 1 pointtype 0
set linestyle 2 pointtype 0
set linestyle 3 pointtype 0
set linestyle 4 pointtype 0
set linestyle 5 pointtype 0
set linestyle 6 pointtype 0
set linestyle 7 pointtype 0
set linestyle 8 pointtype 0
set linestyle 9 pointtype 0

set style fill transparent solid 0.1 noborder

datafile_basename = "'''+datafile_basename+'''"
datafile_name = datafile_basename.".dat"
poinfile_name = datafile_basename."-all_points.dat"

set output datafile_basename.".".datafile_ext

set title "'''+regul_type+'''"
# set title "'''+structure_type+'''-'''+deformation_type+'''-'''+regul_type+'''"

set key right box opaque textcolor variable width -1

set grid

set xlabel "regularization strength"
set xrange ['''+str(regul_level_for_zero)+''':1]
set xtics add ("0" '''+str(regul_level_for_zero)+''')
set xtics add ("0.99" 1e0)
set format x "%g"
set logscale x

set ylabel "normalized displacement error (%)"
set yrange [1e-1:1e+2]
set logscale y

plot ''')
        for k_noise_level,noise_level in enumerate(noise_level_lst):
            plotfile.write((('''     ''')*(k_noise_level>0))+'''datafile_name index '''+str(k_noise_level)+''' using ($2):(100*$3)                        with lines        linestyle '''+str(k_noise_level+1)+''' linewidth 3   title "noise = '''+str(noise_level)+'''"'''+''',\\\n''')
            #plotfile.write(  '''     '''                    +'''datafile_name index '''+str(k_noise_level)+''' using ($2):(100*$3):(100*$4)               with errorbars    linestyle '''+str(k_noise_level+1)+'''             notitle'''                                   +''',\\\n''')
            #plotfile.write(  '''     '''                    +'''datafile_name index '''+str(k_noise_level)+''' using ($2):(100*$3-100*$4):(100*$3+100*$4) with filledcurves linestyle '''+str(k_noise_level+1)+'''             notitle'''                                   +((''',\\
            plotfile.write(  '''     '''                    +'''poinfile_name index '''+str(k_noise_level)+''' using ($2):(100*$3) with points linestyle '''+str(k_noise_level+1)+'''             notitle'''                                   +((''',\\
''')*(k_noise_level<len(noise_level_lst)-1)))

        plotfile.close()

########################################################################

    if (generate_plot):
        os.system("gnuplot "+datafile_basename+".plt")
        os.system("convert -density 300 "+datafile_basename+".pdf"+" "+datafile_basename+".png")

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(plot_disp_error_vs_regul_strength)
