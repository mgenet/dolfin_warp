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
import os

import myVTKPythonLibrary as myvtk

################################################################################

def compute_strain_error(
        k_frame,
        folder,
        noisy,
        working_folder="FEniCS_beta",
        working_basename="Cspamm_normalized-equilibrated",
        working_ext="vtu",
        ref_folder="solution_dt_10msec",
        ref_basename="solution",
        ref_ext="vtk"):

    if (noisy == 1):
        folder = folder + "/ver"
        n_realizations = len(glob.glob(folder+"*"))
        assert (n_realizations), "There is no analysis folder for noisy images. Aborting."
    elif (noisy == 0):
        n_realizations = 1

    for k_realization in range(n_realizations):
        dat_file = folder+"%s/GlobalNormalizedRMSE_forES.dat"%(str(k_realization+1) if noisy else "")

        with open(dat_file, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]

        best_beta = last_line.split()[0]
        beta_best = float(best_beta[12:])

        ref_filenames = glob.glob(ref_folder+"/"+ref_basename+"_[0-9]*."+ref_ext)
        assert (len(ref_filenames) > 0), "There is no working file in the reference mesh folder ("+ref_folder+"/"+ref_basename+"_[0-9]*."+ref_ext+"). Aborting."
        ref_zfill = len(ref_filenames[0].rsplit("_",1)[-1].split(".")[0])

        ref_file = ref_folder+"/"+ref_basename+"_"+str(k_frame).zfill(ref_zfill)+"."+ref_ext
        ref = myvtk.readUGrid(filename=ref_file)
        n_cells = ref.GetNumberOfCells()
        farray_ref_strain = ref.GetCellData().GetArray('Strain_PPS')

        dat_file = folder+"%s/GlobalNormalizedRMSE_forES.dat"%(str(k_realization+1) if noisy else "")

        file = open(folder+"%s/strainerror.dat"%(str(k_realization+1) if noisy else ""), "w")

        working_filenames_beta0 = glob.glob(folder+"%s/"%(str(k_realization+1) if noisy else "")+working_folder+"0/"+working_basename+"_[0-9]*."+working_ext)
        assert (len(working_filenames_beta0) > 0), "There is no working file in the analysis folder with beta = 0. Aborting."
        working_zfill = len(working_filenames_beta0[0].rsplit("_",1)[-1].split(".")[0])
        beta_0_mesh_file = folder+"%s/"%(str(k_realization+1) if noisy else "")+working_folder+"0/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext

        working_filenames_betabest = glob.glob(folder+"%s/"%(str(k_realization+1) if noisy else "")+working_folder+str(beta_best)+"/"+working_basename+"_[0-9]*."+working_ext)
        assert (len(working_filenames_betabest) > 0), "There is no working file in the analysis folder with optimal beta. Aborting."
        working_zfill = len(working_filenames_betabest[0].rsplit("_",1)[-1].split(".")[0])
        beta_best_mesh_file = folder+"%s/"%(str(k_realization+1) if noisy else "")+working_folder+str(beta_best)+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext

        if os.path.exists(beta_0_mesh_file):
            mesh_beta_0 = myvtk.readUGrid(filename=beta_0_mesh_file)
            farray_beta0_strain = mesh_beta_0.GetCellData().GetArray('Strain_PPS')
            diff_radial_beta_0 = numpy.empty(n_cells)
            diff_circumferential_beta_0 = numpy.empty(n_cells)
            diff_longitudinal_beta_0 = numpy.empty(n_cells)

        mesh_beta_best = myvtk.readUGrid(filename=beta_best_mesh_file)
        farray_betaBest_strain = mesh_beta_best.GetCellData().GetArray('Strain_PPS')
        diff_radial_beta_best = numpy.empty(n_cells)
        diff_circumferential_beta_best = numpy.empty(n_cells)
        diff_longitudinal_beta_best = numpy.empty(n_cells)

        for k_cell in range(n_cells):
            if os.path.exists(beta_0_mesh_file):
                diff_radial_beta_0[k_cell] = farray_beta0_strain.GetTuple(k_cell)[0]-farray_ref_strain.GetTuple(k_cell)[0]
                diff_circumferential_beta_0[k_cell] = farray_beta0_strain.GetTuple(k_cell)[1]-farray_ref_strain.GetTuple(k_cell)[1]
                diff_longitudinal_beta_0[k_cell] = farray_beta0_strain.GetTuple(k_cell)[2]-farray_ref_strain.GetTuple(k_cell)[2]

            diff_radial_beta_best[k_cell] = farray_betaBest_strain.GetTuple(k_cell)[0]-farray_ref_strain.GetTuple(k_cell)[0]
            diff_circumferential_beta_best[k_cell] = farray_betaBest_strain.GetTuple(k_cell)[1]-farray_ref_strain.GetTuple(k_cell)[1]
            diff_longitudinal_beta_best[k_cell] = farray_betaBest_strain.GetTuple(k_cell)[2]-farray_ref_strain.GetTuple(k_cell)[2]

        if os.path.exists(beta_0_mesh_file):
            file.write(str(numpy.mean(diff_radial_beta_0))+" "+str(numpy.std(diff_radial_beta_0))+"\n")
            file.write(str(numpy.mean(diff_circumferential_beta_0))+" "+str(numpy.std(diff_circumferential_beta_0))+"\n")
            file.write(str(numpy.mean(diff_longitudinal_beta_0))+" "+str(numpy.std(diff_longitudinal_beta_0))+"\n")

        file.write(str(numpy.mean(diff_radial_beta_best))+" "+str(numpy.std(diff_radial_beta_best))+"\n")
        file.write(str(numpy.mean(diff_circumferential_beta_best))+" "+str(numpy.std(diff_circumferential_beta_best))+"\n")
        file.write(str(numpy.mean(diff_longitudinal_beta_best))+" "+str(numpy.std(diff_longitudinal_beta_best))+"\n")

        file.close()
