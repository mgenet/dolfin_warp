#coding=utf8

########################################################################
###                                                                  ###
### Created by Ezgi Berberoglu, 2017-2021                            ###
###                                                                  ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

from os import path
import numpy as np
import glob
import math

import compute_displacement_infinity_norm as inf_norm
import compute_displacement_error_field as rmse

def compute_best_beta(
    k_frame,
    betas,
    folder,
    noisy,
    disp_array_name = "displacement",
    working_folder = "FEniCS_beta",
    working_basename = "Cspamm_normalized-equilibrated",
    working_ext="vtu",
    ref_mesh_folder="solution_dt_10msec",
    ref_mesh_basename="solution",
    ref_mesh_ext="vtk"):

    if noisy == 1:
        folder = folder + "/ver"
        n_realizations = len(glob.glob("./"+folder+"*"))
        assert (n_realizations), "There is no analysis folder for noisy images. Aborting."
    elif noisy == 0:
        n_realizations = 1

    infNorm = inf_norm.compute_displacement_infinity_norm(
        k_frame = k_frame,
        disp_array_name = disp_array_name,
        ref_mesh_folder = ref_mesh_folder,
        ref_mesh_basename = ref_mesh_basename,
        ref_mesh_ext = ref_mesh_ext)

    for k_realization in range(n_realizations):
        minimum = float("+Inf")
        filename_min = " "
        min_disp_diff = []

        file = open("./"+folder+"%s/globalNormalizedRMSE_forES.dat" %(str(k_realization+1) if noisy else ""), "w")

        for k_beta in range(len(betas)):
            disp_diff = rmse.compute_displacement_error_field(
                k_frame = k_frame,
                disp_array_name = disp_array_name,
                working_folder = "./"+folder+"%s/" %(str(k_realization+1) if noisy else "") +working_folder+betas[k_beta],
                working_basename = working_basename,
                working_ext = working_ext,
                ref_mesh_folder = ref_mesh_folder,
                ref_mesh_basename = ref_mesh_basename,
                ref_mesh_ext = ref_mesh_ext)

            mean_error_norm = np.mean(disp_diff)/infNorm

            if mean_error_norm < minimum:
                minimum = mean_error_norm
                filename_min = working_folder+"_"+betas[k_beta]
                min_disp_diff = disp_diff

            file.write("Case: " + working_folder + "_" + betas[k_beta] + " " + str(mean_error_norm) + " and std: " + str(np.std(disp_diff)/infNorm) + "\n")

        file.write(filename_min + " has the mininum error: " + str(minimum) + " and std: " + str(np.std(min_disp_diff)/infNorm))

        file.close()
