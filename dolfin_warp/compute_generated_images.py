#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_generated_images(
        working_folder                                   ,
        working_basename                                 ,
        working_ext                     = "vtu"          ,
        working_displacement_field_name = "displacement" ,
        ref_image                       = None           ,
        ref_image_folder                = None           ,
        ref_image_basename              = None           ,
        ref_image_ext                   = "vti"          ,
        ref_frame                       = 0              ,
        texture_type                    = "tagging"      ,
        resampling_factor               = 1              ,
        quadrature_degree               = 1              ,
        noise_params                    = {"type":"no"}  ,
        suffix                          = "generated"    ,
        verbose                         = 0              ):

    assert (( ref_image          is not None )
         or ((ref_image_folder   is not None)
         and (ref_image_basename is not None))), "Must provide ref_image or ref_image_folder and ref_image_basename. Aborting."

    if ((ref_image_folder   is not None)
    and (ref_image_basename is not None)):
        ref_image_series = dwarp.ImageSeries(
            folder=ref_image_folder,
            basename=ref_image_basename,
            ext=ref_image_ext)
        ref_image_filename = ref_image_series.get_image_filename(k_frame=0)
    else:
        ref_image_filename = working_folder + "compute_generated_images.vti"
        myvtk.writeImage(ref_image, ref_image_filename)

    working_series = dwarp.MeshSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    mesh_filename  = working_folder
    mesh_filename += "/"+working_basename
    mesh_filename += "-"+"mesh"
    mesh_filename += "."+"xml"
    mesh = dolfin.Mesh(mesh_filename)
    im_dim = mesh.geometry().dim()

    problem = dwarp.FullKinematicsWarpingProblem(
        working_folder=working_folder,
        working_basename=working_basename,
        mesh=mesh,
        print_out=bool(verbose))

    if ((ref_image_folder   is not None)
    and (ref_image_basename is not None)):
        ref_image_series = dwarp.ImageSeries(
            folder=ref_image_folder,
            basename=ref_image_basename,
            ext=ref_image_ext)
    else:
        ref_image_series = dwarp.ImageSeries(
            folder=working_folder,
            basename="compute_generated_images",
            ext="vti")

    energy = dwarp.GeneratedImageContinuousEnergy(
        problem=problem,
        image_series=ref_image_series,
        quadrature_degree=quadrature_degree,
        texture=texture_type,
        resampling_factor=resampling_factor,
        w_char_func=False)

    noise = dwarp.Noise(params=noise_params)

    for k_frame in range(working_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        ugrid = working_series.get_mesh(k_frame=k_frame)
        array_U = ugrid.GetPointData().GetArray(working_displacement_field_name)
        array_U = vtk.util.numpy_support.vtk_to_numpy(array_U)
        array_U = array_U[:,:im_dim]
        array_U = numpy.reshape(array_U, array_U.size)
        problem.U.vector().set_local(array_U)
        problem.U.vector().apply("insert")

        energy.IDIgen.update_disp()
        energy.IDIgen.update_generated_image()
        
        image_filename = working_series.get_mesh_filename(k_frame=k_frame, suffix=suffix, ext="vti")
        energy.IDIgen.write_image(
            image_name="generated",
            filename=image_filename)

        if (noise_params.get("type", "no") != "no"):
            image = myvtk.readImage(image_filename)
            scalars = image.GetPointData().GetScalars()
            I = numpy.empty(1)
            for k_point in range(image.GetNumberOfPoints()):
                scalars.GetTuple(k_point, I)
                noise.add_noise(I)
                scalars.SetTuple(k_point, I)
            myvtk.writeImage(image, image_filename)
