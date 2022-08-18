#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy
import vtk

import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_displacements_from_ref(
        working_folder,
        working_basename,
        ref_frame,
        working_ext="vtu",
        suffix=None,
        verbose=0):

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)
    if (verbose): print("n_frames = "+str(working_series.n_frames))

    ref_mesh = working_series.get_mesh(k_frame=ref_frame)
    n_points = ref_mesh.GetNumberOfPoints()
    n_cells = ref_mesh.GetNumberOfCells()

    ref_disp_farray = myvtk.createDoubleArray(name="ref_disp")
    ref_disp_farray.DeepCopy(ref_mesh.GetPointData().GetVectors())

    warper = vtk.vtkWarpVector()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        warper.SetInputData(ref_mesh)
    else:
        warper.SetInput(ref_mesh)
    warper.Update()
    warped_mesh = warper.GetOutput()
    warped_disp_farray = warped_mesh.GetPointData().GetVectors()

    for k_frame in range(working_series.n_frames):
        cur_mesh = working_series.get_mesh(k_frame=k_frame)
        cur_disp_farray = cur_mesh.GetPointData().GetVectors()
        [warped_disp_farray.SetTuple(
            k_point,
            numpy.substract(
                cur_disp_farray.GetTuple(k_point),
                ref_disp_farray.GetTuple(k_point))) for k_point in range(n_points)]
        myvtk.writeUGrid(
            ugrid=warped_mesh,
            filename=working_series.get_mesh_filename(k_frame=ref_frame, suffix=suffix),
            verbose=verbose)
