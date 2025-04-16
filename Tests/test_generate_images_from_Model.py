#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import os
import shutil
import sys

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk
import dolfin_mech        as dmech
import dolfin_warp        as dwarp

################################################################################

res_folder = sys.argv[0][:-3]
if not os.path.exists(res_folder): os.mkdir(res_folder)

# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=0,
#     clean_after_tests=1)

working_basename  = "heart"

dmech.run_HeartSlice_Hyperelasticity(
    incomp                                 = 0,
    mesh_params                            = {"X0":0.5, "Y0":0.5, "Ri":0.2, "Re":0.4, "l":0.1, "mesh_filebasename":res_folder+"/"+working_basename+"-mesh"},
    mat_params                             = {"model":"CGNH", "parameters":{"E":1., "nu":0.3}},
    step_params                            = {"dt_ini":1/10},
    load_params                            = {"type":"disp", "dRi":-0.10, "dRe":-0.05, "dTi":-math.pi/4, "dTe":-math.pi/8},
    res_basename                           = res_folder+"/"+working_basename,
    write_vtus_with_preserved_connectivity = True,
    verbose                                = 1)

ref_image = myvtk.createImageFromSizeAndRes(
    dim  = 2 ,
    size = 1.,
    res  = 10,
    up   = 1 )

s = [0.1]*2
ref_image_model = lambda X:math.sqrt(abs(math.sin(math.pi*X[0]/s[0]))
                                   * abs(math.sin(math.pi*X[1]/s[1])))

noise_params = {"type":"no"}

dwarp.compute_warped_images(
    working_folder                  = res_folder,
    working_basename                = working_basename,
    working_ext                     = "vtu",
    working_displacement_field_name = "U",
    ref_image                       = ref_image,
    ref_frame                       = 0,
    ref_image_model                 = ref_image_model,
    noise_params                    = noise_params,
    print_warped_mesh               = 0,
    verbose                         = 0)

shutil.rmtree(res_folder, ignore_errors=1)
