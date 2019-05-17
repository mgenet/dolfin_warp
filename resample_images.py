#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import math
import numpy
import os
import random
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

import dolfin_dic as ddic

################################################################################

def resample_images(
        images_folder,
        images_basename,
        resampling_factors,
        images_ext="vti",
        write_temp_images=False):

    images_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    images_nframes = len(images_filenames)
    images_zfill = len(images_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    image_filename = images_folder+"/"+images_basename+"_"+str(0).zfill(images_zfill)+"."+images_ext
    image = myvtk.readImage(
        filename=image_filename,
        verbose=0)
    images_npoints = image.GetNumberOfPoints()
    images_nvoxels = myvtk.getImageDimensions(
        image=image,
        verbose=0)
    images_ndim = len(images_nvoxels)

    if   (images_ext == "vtk"):
        reader = vtk.vtkImageReader()
        writer = vtk.vtkImageWriter()
        if (write_temp_images):
            writer_fft  = vtk.vtkImageWriter()
            writer_mult = vtk.vtkImageWriter()
    elif (images_ext == "vti"):
        reader = vtk.vtkXMLImageDataReader()
        writer = vtk.vtkXMLImageDataWriter()
        if (write_temp_images):
            writer_fft  = vtk.vtkXMLImageDataWriter()
            writer_mult = vtk.vtkXMLImageDataWriter()
    else:
        assert 0, "\"ext\" must be \".vtk\" or \".vti\". Aborting."

    fft = vtk.vtkImageFFT()
    fft.SetDimensionality(images_ndim)
    fft.SetInputConnection(reader.GetOutputPort())
    if (write_temp_images): writer_fft.SetInputConnection(fft.GetOutputPort())

    image_filename = images_folder+"/"+images_basename+"_"+str(0).zfill(images_zfill)+"."+images_ext
    mask_image = myvtk.readImage(
        filename=image_filename,
        verbose=0)
    mask_scalars = myvtk.createDoubleArray(
        name="ImageScalars",
        n_components=2,
        n_tuples=images_npoints,
        verbose=0)
    mask_image.GetPointData().SetScalars(mask_scalars)
    # print mask_image.GetScalarType()
    # print mask_image.GetPointData().GetScalars()
    if (images_ndim == 1):
        for k_point in xrange(images_npoints):
            if ((k_point >                     images_nvoxels[0]/resampling_factors[0]/2) \
            and (k_point < images_nvoxels[0] - images_nvoxels[0]/resampling_factors[0]/2)):
                mask_scalars.SetTuple(k_point, [0, 0])
            else:
                mask_scalars.SetTuple(k_point, [1, 1])
    if (images_ndim == 2):
        for k_point in xrange(images_npoints):
            # print "k_point = "+str(k_point)
            k_y = math.floor(k_point/images_nvoxels[0])
            k_x = k_point - k_y*images_nvoxels[0]
            # print "k_x = "+str(k_x)
            # print "k_y = "+str(k_y)
            if (((k_x >                     images_nvoxels[0]/resampling_factors[0]/2)  \
            and  (k_x < images_nvoxels[0] - images_nvoxels[0]/resampling_factors[0]/2)) \
            or  ((k_y >                     images_nvoxels[1]/resampling_factors[1]/2)  \
            and  (k_y < images_nvoxels[1] - images_nvoxels[1]/resampling_factors[1]/2))):
                mask_scalars.SetTuple(k_point, [0, 0])
            else:
                mask_scalars.SetTuple(k_point, [1, 1])
    if (images_ndim == 3):
        for k_point in xrange(images_npoints):
            k_z = math.floor(k_point/images_nvoxels[0]/images_nvoxels[1])
            k_y = math.floor((k_point - k_z*images_nvoxels[0]*images_nvoxels[1])/images_nvoxels[0])
            k_x = k_point - k_z*images_nvoxels[0]*images_nvoxels[1] - k_y*images_nvoxels[0]
            if (((k_x >                     images_nvoxels[0]/resampling_factors[0]/2)  \
            and  (k_x < images_nvoxels[0] - images_nvoxels[0]/resampling_factors[0]/2)) \
            or  ((k_y >                     images_nvoxels[1]/resampling_factors[1]/2)  \
            and  (k_y < images_nvoxels[1] - images_nvoxels[1]/resampling_factors[1]/2)) \
            or  ((k_z >                     images_nvoxels[2]/resampling_factors[2]/2)  \
            and  (k_z < images_nvoxels[2] - images_nvoxels[2]/resampling_factors[2]/2))):
                mask_scalars.SetTuple(k_point, [0, 0])
            else:
                mask_scalars.SetTuple(k_point, [1, 1])
    if (write_temp_images): myvtk.writeImage(
        image=mask_image,
        filename=images_folder+"/"+images_basename+"_mask"+"."+images_ext,
        verbose=0)

    mult = vtk.vtkImageMathematics()
    mult.SetOperationToMultiply()
    mult.SetInputData(0, mask_image)
    mult.SetInputConnection(1, fft.GetOutputPort())
    if (write_temp_images): writer_mult.SetInputConnection(mult.GetOutputPort())

    rfft = vtk.vtkImageRFFT()
    rfft.SetDimensionality(images_ndim)
    rfft.SetInputConnection(mult.GetOutputPort())

    writer.SetInputConnection(rfft.GetOutputPort())

    for k_frame in xrange(images_nframes):
        reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        reader.Update()

        fft.Update()
        if (write_temp_images):
            writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer_fft.Update()
            writer_fft.Write()
        # print fft.GetOutput().GetScalarType()
        # print fft.GetOutput().GetPointData().GetScalars()

        mult.Update()
        if (write_temp_images):
            writer_mult.SetFileName(images_folder+"/"+images_basename+"_mult"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer_mult.Update()
            writer_mult.Write()

        rfft.Update()
        writer.SetFileName(images_folder+"/"+images_basename+"_resampled"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        writer.Update()
        writer.Write()
