#coding=utf8

########################################################################

import numpy
import os

import myPythonLibrary as mypy
import dolfin_warp     as dwarp

########################################################################

def plot_disp_error_vs_mesh_size(
        images_folder                            : str          ,
        sol_folder                               : str          ,
        structure_type                           : str          ,
        deformation_type                         : str          ,
        texture_type                             : str          ,
        regul_type                               : str          ,
        noise_level_lst                          : list         ,
        n_runs_for_noisy_images                  : int          ,
        regul_level_lst                          : list         ,
        mesh_size_lst                            : list         ,
        with_refine                              : bool  = False,
        error_for_nan                            : float = None ,
        generate_datafile                        : bool  = True ,
        generate_datafile_with_limited_precision : bool  = False,
        generate_plotfile                        : bool  = True ,
        generate_plot                            : bool  = True ):

    print ("structure_type:"  , structure_type  )
    print ("deformation_type:", deformation_type)
    print ("texture_type:"    , texture_type    )
    print ("regul_type:"      , regul_type      )

    script_basename = "plot_disp_error_vs_mesh_size"

    if not os.path.exists(script_basename):
        os.mkdir(script_basename)
    
    datafile_basename  = script_basename
    datafile_basename += "/"+structure_type
    datafile_basename += "-"+deformation_type
    datafile_basename += "-"+texture_type
    datafile_basename += "-"+regul_type
    if (with_refine):
        datafile_basename += "-with_refine"

########################################################################

    if (generate_datafile) or (generate_datafile_with_limited_precision):

        if (generate_datafile): data_printer = mypy.DataPrinter(
            names=["noise_level", "regul_level", "mesh_size", "disp_err_avg", "disp_err_std", "disp_err_min", "disp_err_max"],
            filename=datafile_basename+".dat",
            limited_precision=False)

        if (generate_datafile_with_limited_precision): data_printer2 = mypy.DataPrinter(
            names=["noise_level", "regul_level", "mesh_size", "disp_err_avg", "disp_err_std", "disp_err_min", "disp_err_max"],
            filename=datafile_basename+"-limited_precision.dat",
            limited_precision=True)

        if (generate_datafile): data_printer3 = mypy.DataPrinter(
            names=["noise_level", "regul_level", "mesh_size", "disp_err"],
            filename=datafile_basename+"-all_points.dat",
            limited_precision=False)

        if (structure_type in ("square", "disc", "ring")):
            ref_disp_array_name = "displacement"
        elif (structure_type in ("heart")):
            ref_disp_array_name = "U"
        else: assert (0)

        for noise_level in noise_level_lst:
         for regul_level in regul_level_lst:
          for k_mesh_size, mesh_size in enumerate(mesh_size_lst):

            print ("noise_level:", noise_level)
            print ("regul_level:", regul_level)
            print ("mesh_size:"  , mesh_size  )

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

                sol_basename  = images_basename
                if not (with_refine):
                    sol_basename += "-h="+str(mesh_size)
                sol_basename += "-"+regul_type
                sol_basename += "-regul="+str(regul_level)
                if (with_refine):
                    sol_basename += "-refine="+str(k_mesh_size)

                # import time
                # t = time.time()
                # disp_err_vtk = dwarp.compute_displacement_error_with_vtk(
                #     working_folder=sol_folder,
                #     working_basename=sol_basename,
                #     working_disp_array_name="displacement",
                #     ref_folder=images_folder,
                #     ref_basename=structure_type+"-"+deformation_type+"-h="+str(mesh_size),
                #     ref_disp_array_name=ref_disp_array_name,
                #     verbose=0)
                # print ("t (vtk)", time.time() - t)
                # print ("disp_err_vtk:", disp_err_vtk)
                # t = time.time()
                # disp_err_np = dwarp.compute_displacement_error_with_numpy(
                #     working_folder=sol_folder,
                #     working_basename=sol_basename,
                #     working_disp_array_name="displacement",
                #     ref_folder=images_folder,
                #     ref_basename=structure_type+"-"+deformation_type+"-h="+str(mesh_size),
                #     ref_disp_array_name=ref_disp_array_name,
                #     verbose=0)
                # print ("t (np)", time.time() - t)
                # print ("disp_err_np:", disp_err_np)
                # assert (disp_err_np == disp_err_vtk)
                # t = time.time()
                # disp_err_fenics = dwarp.compute_displacement_error_with_fenics(
                #     working_folder=sol_folder,
                #     working_basename=sol_basename,
                #     working_disp_array_name="displacement",
                #     ref_folder=images_folder,
                #     ref_basename=structure_type+"-"+deformation_type,
                #     ref_disp_array_name=ref_disp_array_name,
                #     verbose=0)
                # print ("t (fenics)", time.time() - t)
                # print ("disp_err_fenics:", disp_err_fenics)
                # disp_err = disp_err_fenics

                try:
                    disp_err = dwarp.compute_displacement_error_with_fenics(
                        working_folder=sol_folder,
                        working_basename=sol_basename,
                        working_disp_array_name="displacement",
                        ref_folder=images_folder,
                        ref_basename=structure_type+"-"+deformation_type,
                        ref_disp_array_name=ref_disp_array_name,
                        verbose=0)
                    # disp_err = dwarp.compute_displacement_error(
                    #     working_folder=sol_folder,
                    #     working_basename=sol_basename,
                    #     working_disp_array_name="displacement",
                    #     ref_folder=images_folder,
                    #     ref_basename=structure_type+"-"+deformation_type+"-h="+str(mesh_size),
                    #     ref_disp_array_name=ref_disp_array_name,
                    #     verbose=0)
                except:
                    disp_err = error_for_nan
                print ("disp_err:", disp_err)

                disp_err_lst.append(disp_err)

                if (error_for_nan is not None):
                    if numpy.isnan(disp_err) or (disp_err > error_for_nan):
                        disp_err = error_for_nan
                if (generate_datafile): data_printer3.write_line([noise_level, regul_level, mesh_size, disp_err])

            disp_err_avg = numpy.mean(disp_err_lst)
            disp_err_std = numpy.std(disp_err_lst)
            disp_err_min = numpy.min(disp_err_lst)
            disp_err_max = numpy.max(disp_err_lst)
            print ("disp_err_avg:", disp_err_avg)
            print ("disp_err_std:", disp_err_std)
            print ("disp_err_min:", disp_err_min)
            print ("disp_err_max:", disp_err_max)

            if (error_for_nan is not None):
                if numpy.isnan(disp_err_avg) or (disp_err_avg > error_for_nan):
                    disp_err_avg = error_for_nan
                    disp_err_std = 0.
                    if (disp_err_min > error_for_nan):
                        disp_err_min = error_for_nan
                    disp_err_max = error_for_nan

            if (generate_datafile                       ): data_printer.write_line([noise_level, regul_level, mesh_size, disp_err_avg, disp_err_std, disp_err_min, disp_err_max])
            if (generate_datafile_with_limited_precision): data_printer2.write_line([noise_level, regul_level, mesh_size, disp_err_avg, disp_err_std, disp_err_min, disp_err_max])

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
set terminal pdf enhanced size 5,3; outputfile_ext = "pdf"

load "Set1.plt"

set linestyle  1 pointtype 0
set linestyle  2 pointtype 0
set linestyle  3 pointtype 0
set linestyle  4 pointtype 0
set linestyle  5 pointtype 0
set linestyle  6 pointtype 0
set linestyle  7 pointtype 0
set linestyle  8 pointtype 0
set linestyle  9 pointtype 0
set linestyle 10 pointtype 0
set linestyle 11 pointtype 0
set linestyle 12 pointtype 0

set style fill transparent solid 0.1 noborder

datafile_basename = "'''+datafile_basename+'''"
datafile_name = datafile_basename.".dat"
poinfile_name = datafile_basename."-all_points.dat"

set output datafile_basename.".".outputfile_ext

set title "'''+structure_type+'''-'''+deformation_type+'''-'''+regul_type+'''"

set key outside right center box textcolor variable width -3

set grid

set xlabel "mesh size"
set xrange [0.1/2**4:0.1]
# set format x "%g"
set logscale x

set ylabel "normalized displacement error (%)"
set yrange [1e-1:1e+3]
set logscale y

plot ''')
        for k_noise_level,noise_level in enumerate(noise_level_lst):
         for k_regul_level,regul_level in enumerate(regul_level_lst):
            if   (noise_level == 0.0): lc = 1
            elif (noise_level == 0.1): lc = 2
            elif (noise_level == 0.2): lc = 3
            elif (noise_level == 0.3): lc = 4
            else: assert (0)
            if   (regul_level == 0.0 ): dt = 1
            elif (regul_level == 0.1 ): dt = 2
            elif (regul_level == 0.99): dt = 3
            else: assert (0)
            # index_for_plot = k_regul_level*len(noise_level_lst)+k_noise_level+1
            index_for_data = k_noise_level*len(regul_level_lst)+k_regul_level
            plotfile.write((('''     ''')*((k_noise_level>0)or(k_regul_level>0)))+'''datafile_name index '''+str(index_for_data)+''' using ($3):(100*$4)                        with lines        linestyle '''+str(lc)+''' dashtype '''+str(dt)+''' linewidth 3   title "noise = '''+"{:1.1f}".format(noise_level)+''', regul = '''+"{:1.2f}".format(regul_level)+'''"'''+''',\\\n''')
            if (noise_level == 0): continue
            # plotfile.write(  '''     '''                                         +'''datafile_name index '''+str(index_for_data)+''' using ($3):(100*$4):(100*$5)               with errorbars    linestyle '''+str(lc)+'''             notitle'''                                   +''',\\\n''')
            # plotfile.write(  '''     '''                                         +'''datafile_name index '''+str(index_for_data)+''' using ($3):(100*$4-100*$5):(100*$4+100*$5) with filledcurves linestyle '''+str(lc)+'''             notitle'''                                   +((''',\\
            plotfile.write(  '''     '''                    +'''poinfile_name index '''+str(index_for_data)+''' using ($3):(100*$4) with points linestyle '''+str(lc)+'''             notitle'''                                   +((''',\\
''')*((k_noise_level<len(noise_level_lst)-1)or(k_regul_level<len(regul_level_lst)-1))))

        plotfile.close()

########################################################################

    if (generate_plot):

        os.system("gnuplot "+datafile_basename+".plt")
        os.system("convert -density 300 "+datafile_basename+".pdf"+" "+datafile_basename+".png")

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(plot_disp_error_vs_mesh_size)
