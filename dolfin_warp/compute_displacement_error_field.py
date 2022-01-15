#coding=utf8

########################################################################
###                                                                  ###
### Created by Ezgi Berberoğlu, 2017-2021                            ###
###                                                                  ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

from builtins import range

import glob
import math
import numpy

import myVTKPythonLibrary as myvtk

################################################################################

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

    working_mesh = myvtk.readUGrid(
        filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext,
        verbose=0)
    n_points = working_mesh.GetNumberOfPoints()

    ref_mesh = myvtk.readUGrid(
        filename=ref_mesh_folder+"/"+ref_mesh_basename+"_"+str(k_frame).zfill(working_zfill)+"."+ref_mesh_ext,
        verbose=0)

    assert (ref_mesh.GetNumberOfPoints() == n_points),\
        "Reference and working meshes should have the same number of points. Aborting."

    farray_U_ref = ref_mesh.GetPointData().GetArray(disp_array_name)
    farray_U = working_mesh.GetPointData().GetArray(disp_array_name)

    disp_diff = numpy.empty(n_points)

    for k_point in range(n_points):
        disp_diff[k_point] = math.sqrt(numpy.sum(numpy.square(numpy.subtract(
            farray_U_ref.GetTuple(k_point),
            farray_U.GetTuple(k_point)))))

    return disp_diff
