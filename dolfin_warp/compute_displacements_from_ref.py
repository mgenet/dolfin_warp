#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2021                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import glob
import numpy
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_displacements_from_ref(
        working_folder,
        working_basename,
        ref_frame,
        working_ext="vtu",
        suffix="",
        verbose=0):

    working_filenames = glob.glob(working_folder+"/"+working_basename+"_[0-9]*."+working_ext)
    working_zfill = len(working_filenames[0].rsplit("_",1)[-1].split(".")[0])
    n_frames = len(working_filenames)
    if (verbose): print("n_frames = "+str(n_frames))

    ref_mesh = myvtk.readUGrid(
        filename=working_folder+"/"+working_basename+"_"+str(ref_frame).zfill(working_zfill)+"."+working_ext,
        verbose=verbose)
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

    for k_frame in range(n_frames):
        cur_mesh = myvtk.readUGrid(
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext,
            verbose=verbose)
        cur_disp_farray = cur_mesh.GetPointData().GetVectors()
        [warped_disp_farray.SetTuple(
            k_point,
            numpy.substract(
                cur_disp_farray.GetTuple(k_point),
                ref_disp_farray.GetTuple(k_point))) for k_point in range(n_points)]
        myvtk.writeUGrid(
            ugrid=warped_mesh,
            filename=working_folder+"/"+working_basename+("-"+suffix)*(suffix!="")+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext,
            verbose=verbose)
