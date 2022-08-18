#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2017-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import numpy

import dolfin_warp as dwarp

################################################################################

def compute_best_regularization_strength(
        k_frame,
        betas,
        folder,
        noisy,
        disp_array_name = "displacement",
        working_folder = "FEniCS_beta",
        working_basename = "Cspamm_normalized-equilibrated",
        working_ext="vtu",
        ref_folder="solution_dt_10msec",
        ref_basename="solution",
        ref_ext="vtk"):

    if (noisy):
        folder = folder + "/ver"
        n_realizations = len(glob.glob(folder+"*"))
        assert (n_realizations), "There is no analysis folder for noisy images. Aborting."
    else:
        n_realizations = 1

    inf_norm = dwarp.compute_displacement_infinity_norm(
        k_frame=k_frame,
        disp_array_name=disp_array_name,
        working_folder=ref_folder,
        working_basename=ref_basename,
        working_ext=ref_ext)

    for k_realization in range(n_realizations):
        minimum = float("+Inf")
        filename_min = " "
        min_disp_diff = []

        file = open(folder+"%s/globalNormalizedRMSE_forES.dat"%(str(k_realization+1) if noisy else ""), "w")

        for k_beta in range(len(betas)):
            disp_diff = dwarp.compute_displacement_error_field(
                k_frame=k_frame,
                disp_array_name=disp_array_name,
                working_folder=folder+"%s/"%(str(k_realization+1) if noisy else "")+working_folder+betas[k_beta],
                working_basename=working_basename,
                working_ext=working_ext,
                ref_folder=ref_folder,
                ref_basename=ref_basename,
                ref_ext=ref_ext)

            mean_error_norm = numpy.mean(disp_diff)/inf_norm

            if (mean_error_norm < minimum):
                minimum = mean_error_norm
                filename_min = working_folder+"_"+betas[k_beta]
                min_disp_diff = disp_diff

            file.write("Case: "+working_folder+"_"+betas[k_beta]+" "+str(mean_error_norm)+" and std: "+str(numpy.std(disp_diff)/inf_norm)+"\n")

        file.write(filename_min+" has the mininum error: "+str(minimum)+" and std: "+str(numpy.std(min_disp_diff)/inf_norm))

        file.close()
