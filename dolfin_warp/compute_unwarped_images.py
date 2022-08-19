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
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_unwarped_images(
        images_folder,
        images_basename,
        working_folder,
        working_basename,
        images_ext="vti",
        working_ext="vtu",
        suffix="unwarped",
        verbose=0):

    images_series = dwarp.ImagesSeries(
        folder=images_folder,
        basename=images_basename,
        ext=images_ext)
    ref_image = images_series.get_image(k_frame=0)
    image = myvtk.createImage(
        origin=ref_image.GetOrigin(),
        spacing=ref_image.GetSpacing(),
        extent=ref_image.GetExtent())
    scalars = image.GetPointData().GetScalars()

    working_series = dwarp.MeshesSeries(
        folder=working_folder,
        basename=working_basename,
        ext=working_ext)

    X = numpy.empty(3)
    U = numpy.empty(3)
    x = numpy.empty(3)
    I = numpy.empty(1)
    m = numpy.empty(1)
    for k_frame in range(working_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        def_image = images_series.get_image(k_frame=k_frame)

        interpolator = myvtk.getImageInterpolator(
            image=def_image)

        mesh = working_series.get_mesh(k_frame=k_frame)

        probe = vtk.vtkProbeFilter()
        if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            probe.SetInputData(image)
            probe.SetSourceData(mesh)
        else:
            probe.SetInput(image)
            probe.SetSource(mesh)
        probe.Update()
        probed_image = probe.GetOutput()
        scalars_mask = probed_image.GetPointData().GetArray("vtkValidPointMask")
        scalars_U = probed_image.GetPointData().GetArray("displacement")

        for k_point in range(image.GetNumberOfPoints()):
            scalars_mask.GetTuple(k_point, m)
            if (m[0] == 0):
                I[0] = 0.
            else:
                image.GetPoint(k_point, X)
                scalars_U.GetTuple(k_point, U)
                x = X + U
                interpolator.Interpolate(x, I)
            scalars.SetTuple(k_point, I)

        myvtk.writeImage(
            image=image,
            filename=images_series.get_mesh_filename(k_frame=k_frame, suffix=suffix))
