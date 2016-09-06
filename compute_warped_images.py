#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2016                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import glob
import numpy
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

########################################################################

def compute_warped_images(
        ref_image_folder,
        ref_image_basename,
        sol_folder,
        sol_basename,
        ref_frame=0,
        sol_ext="vtu",
        verbose=0):

    ref_image_zfill = len(glob.glob(ref_image_folder+"/"+ref_image_basename+"_*.vti")[0].rsplit("_")[-1].split(".")[0])
    ref_image_filename = ref_image_folder+"/"+ref_image_basename+"_"+str(ref_frame).zfill(ref_image_zfill)+".vti"
    ref_image = myvtk.readImage(
        filename=ref_image_filename)

    interpolator = myvtk.getImageInterpolator(
        image=ref_image)
    #I = numpy.empty(1)
    #interpolator.Interpolate([0.35, 0.25, 0.], I)

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

    sol_zfill = len(glob.glob(sol_folder+"/"+sol_basename+"_*."+sol_ext)[0].rsplit("_")[-1].split(".")[0])
    n_frames = len(glob.glob(sol_folder+"/"+sol_basename+"_"+"[0-9]"*sol_zfill+"."+sol_ext))
    #n_frames = 1

    X = numpy.empty(3)
    U = numpy.empty(3)
    x = numpy.empty(3)
    I = numpy.empty(1)
    m = numpy.empty(1)
    for k_frame in xrange(n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        mesh = myvtk.readUGrid(
            filename=sol_folder+"/"+sol_basename+"_"+str(k_frame).zfill(sol_zfill)+"."+sol_ext)
        #print mesh

        warp = vtk.vtkWarpVector()
        if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            warp.SetInputData(mesh)
        else:
            warp.SetInput(mesh)
        warp.Update()
        warped_mesh = warp.GetOutput()
        #myvtk.writeUGrid(
            #ugrid=warped_mesh,
            #filename=sol_folder+"/"+sol_basename+"-warped_"+str(k_frame).zfill(sol_zfill)+"."+sol_ext)

        probe = vtk.vtkProbeFilter()
        if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            probe.SetInputData(image)
            probe.SetSourceData(warped_mesh)
        else:
            probe.SetInput(image)
            probe.SetSource(warped_mesh)
        probe.Update()
        probed_image = probe.GetOutput()
        scalars_mask = probed_image.GetPointData().GetArray("vtkValidPointMask")
        scalars_U = probed_image.GetPointData().GetArray("displacement")
        #myvtk.writeImage(
            #image=probed_image,
            #filename=sol_folder+"/"+sol_basename+"_"+str(k_frame).zfill(sol_zfill)+".vti")

        for k_point in xrange(image.GetNumberOfPoints()):
            scalars_mask.GetTuple(k_point, m)
            if (m[0] == 0):
                I[0] = 0.
            else:
                image.GetPoint(k_point, x)
                scalars_U.GetTuple(k_point, U)
                X = x - U
                interpolator.Interpolate(X, I)
            scalars.SetTuple(k_point, I)

        myvtk.writeImage(
            image=image,
            filename=sol_folder+"/"+sol_basename+"-warped_"+str(k_frame).zfill(sol_zfill)+".vti")
