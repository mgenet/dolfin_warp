#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2012-2022                                       ###
###                                                                          ###
### University of California at San Francisco (UCSF), USA                    ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy
import random
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_warped_images(
        working_folder,
        working_basename,
        working_ext="vtu",
        working_displacement_field_name="displacement",
        ref_image=None,
        ref_image_folder=None,
        ref_image_basename=None,
        ref_image_ext="vti",
        ref_frame=0,
        ref_image_model=None,
        noise_params={},
        suffix="warped",
        print_warped_mesh=0,
        verbose=0):

    assert ((ref_image is not None)
         or ((ref_image_folder   is not None)
         and (ref_image_basename is not None))), "Must provide a ref_image or a ref_image_folder and a ref_image_basename. Aborting."

    if (ref_image is None):
        ref_images_series = dwarp.ImagesSeries(
            folder=ref_image_folder,
            basename=ref_image_basename,
            ext=ref_image_ext)
        ref_image = ref_images_series.get_image(k_frame=ref_frame)

    if (ref_image_model is None):
        ref_image_interpolator = myvtk.getImageInterpolator(
            image=ref_image)

    noise = dwarp.Noise(
        params=noise_params)

    image = myvtk.createImage(
        origin=ref_image.GetOrigin(),
        spacing=ref_image.GetSpacing(),
        extent=ref_image.GetExtent())
    scalars = image.GetPointData().GetScalars()

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    if   (working_ext == "vtk"):
        reader = vtk.vtkUnstructuredGridReader()
    elif (working_ext == "vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.UpdateDataObject()
    ugrid = reader.GetOutput()

    warp = vtk.vtkWarpVector()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        warp.SetInputData(ugrid)
    else:
        warp.SetInput(ugrid)
    warp.UpdateDataObject()
    warped_ugrid = warp.GetOutput()

    probe = vtk.vtkProbeFilter()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        probe.SetInputData(image)
        probe.SetSourceData(warped_ugrid)
    else:
        probe.SetInput(image)
        probe.SetSource(warped_ugrid)
    probe.UpdateDataObject()
    probed_image = probe.GetOutput()

    X = numpy.empty(3)
    U = numpy.empty(3)
    x = numpy.empty(3)
    I = numpy.empty(1)
    m = numpy.empty(1)
    for k_frame in range(working_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(working_series.get_mesh_filename(k_frame=k_frame))
        reader.Update()
        # print(ugrid)

        assert (ugrid.GetPointData().HasArray(working_displacement_field_name)),\
            "no array '" + working_displacement_field_name + "' in ugrid. Aborting."
        ugrid.GetPointData().SetActiveVectors(working_displacement_field_name)
        warp.Update()
        probe.Update()
        scalars_mask = probed_image.GetPointData().GetArray("vtkValidPointMask")
        scalars_U = probed_image.GetPointData().GetArray(working_displacement_field_name)
        if (print_warped_mesh):
            myvtk.writeUGrid(
                ugrid=warped_ugrid,
                filename=working_series.get_mesh_filename(k_frame=k_frame, suffix="warped"))
        #myvtk.writeImage(
            #image=probed_image,
            #filename=working_series.get_mesh_filename(k_frame=k_frame, ext="vti"))

        for k_point in range(image.GetNumberOfPoints()):
            scalars_mask.GetTuple(k_point, m)
            if (m[0] == 0):
                I[0] = 0.
            else:
                image.GetPoint(k_point, x)
                scalars_U.GetTuple(k_point, U)
                X = x - U
                if (ref_image_model is None):
                    ref_image_interpolator.Interpolate(X, I)
                else:
                    I[0] = ref_image_model(X)
            noise.add_noise(I)
            scalars.SetTuple(k_point, I)

        myvtk.writeImage(
            image=image,
            filename=working_series.get_mesh_filename(k_frame=k_frame, suffix=suffix, ext="vti"))
