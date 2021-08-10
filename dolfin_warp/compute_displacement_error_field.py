#coding=utf8

########################################################################
###                                                                  ###
### Created by Ezgi Berberoglu, 2017-2021                            ###
###                                                                  ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import myVTKPythonLibrary as myvtk
import numpy as np
import glob
import math

def compute_displacement_error_field(
    k_frame,
    disp_array_name,
    working_folder,
    working_basename,
    working_ext,
    ref_mesh_folder,
    ref_mesh_basename,
    ref_mesh_ext="vtk"):

    working_filenames = glob.glob(working_folder+"/"+working_basename+"_[0-9]*."+working_ext)
    assert (len(working_filenames) > 0), "There is no working file in the analysis folder. Aborting."
    working_zfill = len(working_filenames[0].rsplit("_",1)[-1].split(".")[0])

    working_mesh = myvtk.readUGrid(filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext, verbose=0)

    ref_mesh = myvtk.readUGrid(
        filename = "./" + ref_mesh_folder + "/" + ref_mesh_basename + "_" + str(k_frame).zfill(working_zfill) + "." + ref_mesh_ext)
    n_points = ref_mesh.GetNumberOfPoints()

    assert n_points == working_mesh.GetNumberOfPoints(), "Reference mesh and the analyzed mesh should have the same geometrical properties. Number of points does not match."

    farray_ref_U = ref_mesh.GetPointData().GetArray(disp_array_name)
    farray_U = working_mesh.GetPointData().GetArray(disp_array_name)

    disp_diff = np.empty(n_points)

    for k_point in range(n_points):
        disp_diff[k_point] = math.sqrt(np.sum(np.square(np.subtract(farray_ref_U.GetTuple(k_point),farray_U.GetTuple(k_point)))))

    return disp_diff
