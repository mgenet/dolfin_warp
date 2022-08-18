#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2017-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import numpy

import dolfin_warp as dwarp

################################################################################

def compute_displacement_error_field(
        working_folder,
        working_basename,
        ref_folder,
        ref_basename,
        k_frame,
        working_ext="vtk",
        ref_ext="vtk",
        disp_array_name="displacement"):

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    working_mesh = working_series.get_mesh(k_frame=k_frame)
    n_points = working_mesh.GetNumberOfPoints()

    ref_series = dwarp.MeshesSeries(
        folder=ref_folder,
        basename=ref_basename,
        ext=ref_ext)

    ref = ref_series.get_mesh(k_frame=k_frame)
    assert (ref.GetNumberOfPoints() == n_points),\
        "Reference and working meshes should have the same number of points. Aborting."

    farray_U_ref = ref.GetPointData().GetArray(disp_array_name)
    farray_U = working_mesh.GetPointData().GetArray(disp_array_name)

    disp_diff = numpy.empty(n_points)

    for k_point in range(n_points):
        disp_diff[k_point] = math.sqrt(numpy.sum(numpy.square(numpy.subtract(
            farray_U_ref.GetTuple(k_point),
            farray_U.GetTuple(k_point)))))

    return disp_diff
