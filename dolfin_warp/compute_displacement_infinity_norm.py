#coding=utf8

################################################################################
###                                                                          ###
### Created by Ezgi Berberoğlu, 2017-2021                                    ###
###                                                                          ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy

import dolfin_warp as dwarp

################################################################################

def compute_displacement_infinity_norm(
        working_folder,
        working_basename,
        k_frame,
        working_ext="vtk",
        disp_array_name="displacement"):

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    working_mesh = working_series.get_mesh(k_frame=k_frame)
    n_points = working_mesh.GetNumberOfPoints()
    farray_U = working_mesh.GetPointData().GetArray(disp_array_name)

    inf_norm = float("-Inf")
    for k_point in range(n_points):
        max = numpy.sqrt(numpy.sum(numpy.square(farray_U.GetTuple(k_point))))
        if (max > inf_norm):
            inf_norm = max

    return inf_norm
