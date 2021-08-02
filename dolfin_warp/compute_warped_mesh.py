#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2012-2021                                       ###
###                                                                          ###
### University of California at San Francisco (UCSF), USA                    ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import range

import dolfin
import numpy
import os
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .generate_images import *

################################################################################

def compute_warped_mesh(
        working_folder,
        working_basename,
        images,
        structure,
        deformation,
        evolution,
        mesh=None,
        mesh_folder=None,
        mesh_basename=None,
        mesh_ext=None,
        verbose=0):

    mypy.my_print(verbose, "*** compute_warped_mesh ***")

    assert ((mesh is not None) or ((mesh_folder is not None) and (mesh_basename is not None) and (mesh_ext is not None))),\
        "Must provide a mesh (mesh = "+str(mesh)+") or a mesh file (mesh_folder = "+str(mesh_folder)+", mesh_basename = "+str(mesh_basename)+", mesh_ext = "+str(mesh_ext)+"). Aborting."

    if (mesh is None):
        mesh_folder       = mesh_folder
        mesh_basename     = mesh_basename
        mesh_filebasename = mesh_folder+"/"+mesh_basename
        mesh_ext          = mesh_ext
        mesh_filename     = mesh_filebasename+"."+mesh_ext
        assert (os.path.exists(mesh_filename)),\
        "No mesh in "+mesh_filename+". Aborting."
        if (mesh_ext == "xml"):
            mesh = dwarp.mesh2ugrid(
                dolfin.Mesh(
                    mesh_filename))
        elif (mesh_ext in ("vtk", "vtu")):
            mesh = myvtk.readDataSet(
                filename=mesh_filename,
                verbose=verbose-1)
    else:
        mesh = dwarp.mesh2ugrid(mesh)
    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    if (mesh_folder is not None) and (mesh_basename is not None) and  os.path.exists(mesh_folder+"/"+mesh_basename+"-WithLocalBasis.vtk"):
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
    for k_frame in range(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mapping.init_t(t)

        for k_point in range(n_points):
            mesh.GetPoint(k_point, X)
            mapping.x(X, x)
            U = x - X
            farray_disp.SetTuple(k_point, U)

        myvtk.addStrainsFromDisplacements(
            mesh=mesh,
            disp_array_name="displacement",
            mesh_w_local_basis=ref_mesh,
            verbose=verbose-1)

        myvtk.writeDataSet(
            dataset=mesh,
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(images["zfill"])+".vtu",
            verbose=verbose-1)
