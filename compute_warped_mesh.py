#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2012-2016                               ###
###                                                                          ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy
import os
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def compute_warped_mesh(
        mesh_folder,
        mesh_basename,
        images,
        structure,
        deformation,
        evolution,
        verbose=0):

    mypy.my_print(verbose, "*** compute_warped_mesh ***")

    mesh = myvtk.readUGrid(
        filename=mesh_folder+"/"+mesh_basename+".vtk",
        verbose=verbose-1)
    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    if os.path.exists(mesh_folder+"/"+mesh_basename+"-WithLocalBasis.vtk"):
        ref_mesh = myvtk.readUGrid(
            filename=mesh_folder+"/"+mesh_basename+"-WithLocalBasis.vtk",
            verbose=verbose-1)
    else:
        ref_mesh = None

    farray_disp = myvtk.createFloatArray(
        name="displacement",
        n_components=3,
        n_tuples=n_points,
        verbose=verbose-1)
    mesh.GetPointData().AddArray(farray_disp)

    mapping = Mapping(images, structure, deformation, evolution)

    X = numpy.empty(3)
    x = numpy.empty(3)
    U = numpy.empty(3)
    if ("zfill" not in images.keys()):
        images["zfill"] = len(str(images["n_frames"]))
    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mapping.init_t(t)

        for k_point in xrange(n_points):
            mesh.GetPoint(k_point, X)
            mapping.x(X, x)
            U = x - X
            farray_disp.SetTuple(k_point, U)

        myvtk.addStrainsFromDisplacements(
            mesh=mesh,
            disp_array_name="displacement",
            mesh_w_local_basis=ref_mesh,
            verbose=verbose-1)

        myvtk.writeUGrid(
            ugrid=mesh,
            filename=mesh_folder+"/"+mesh_basename+"_"+str(k_frame).zfill(images["zfill"])+".vtk",
            verbose=verbose-1)
