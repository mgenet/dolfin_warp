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
import math
import glob

def compute_displacement_infinity_norm(
    k_frame,
    disp_array_name,
    ref_mesh_folder,
    ref_mesh_basename,
    ref_mesh_ext="vtk"):

    ref_filenames = glob.glob(ref_mesh_folder+"/"+ref_mesh_basename+"_[0-9]*."+ref_mesh_ext)
    assert (len(ref_filenames) > 0), "There is no working file in the reference mesh folder ("+working_folder+"/"+working_basename+"_[0-9]*."+working_ext+"). Aborting."

    ref_zfill = len(ref_filenames[0].rsplit("_",1)[-1].split(".")[0])

    ref_mesh_filename = "./" + ref_mesh_folder + "/" + ref_mesh_basename + "_" + str(k_frame).zfill(ref_zfill) + "." + ref_mesh_ext
    ref_mesh = myvtk.readUGrid(
        ref_mesh_filename)

    n_points = ref_mesh.GetNumberOfPoints()
    farray_U = ref_mesh.GetPointData().GetArray(disp_array_name)

    inf_norm = float("-Inf")

    for k_point in range(n_points):
        max =  math.sqrt(np.sum(np.square(farray_U.GetTuple(k_point))))
        if max > inf_norm:
            inf_norm = max

    return inf_norm
