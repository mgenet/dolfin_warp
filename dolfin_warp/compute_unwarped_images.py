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
import glob
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
        working_ext="vtu",
        verbose=0):

    ref_image_zfill = len(glob.glob(images_folder+"/"+images_basename+"_*.vti")[0].rsplit("_")[-1].split(".")[0])
    ref_image_filename = images_folder+"/"+images_basename+"_"+str(0).zfill(ref_image_zfill)+".vti"
    ref_image = myvtk.readImage(
        filename=ref_image_filename)

    image = vtk.vtkImageData()
    image.SetOrigin(ref_image.GetOrigin())
    image.SetSpacing(ref_image.GetSpacing())
    image.SetExtent(ref_image.GetExtent())
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        image.AllocateScalars(vtk.VTK_FLOAT, 1)
    else:
        image.SetScalarTypeToFloat()
        image.SetNumberOfScalarComponents(1)
        image.AllocateScalars()
    scalars = image.GetPointData().GetScalars()

    working_zfill = len(glob.glob(working_folder+"/"+working_basename+"_*."+working_ext)[0].rsplit("_")[-1].split(".")[0])
    n_frames = len(glob.glob(working_folder+"/"+working_basename+"_"+"[0-9]"*working_zfill+"."+working_ext))
    #n_frames = 1

    X = numpy.empty(3)
    U = numpy.empty(3)
    x = numpy.empty(3)
    I = numpy.empty(1)
    m = numpy.empty(1)
    for k_frame in range(n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        def_image = myvtk.readImage(
            filename=images_folder+"/"+images_basename+"_"+str(k_frame).zfill(ref_image_zfill)+".vti")

        interpolator = myvtk.getImageInterpolator(
            image=def_image)

        mesh = myvtk.readUGrid(
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext)

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
            filename=working_folder+"/"+working_basename+"-unwarped_"+str(k_frame).zfill(working_zfill)+".vti")
