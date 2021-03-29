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
import math
import numpy
import os
import random
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_downsampled_images(
        images_folder,
        images_basename,
        downsampling_factors,
        images_ext="vti",
        keep_resolution=0,
        write_temp_images=0,
        suffix="",
        verbose=0):

    mypy.my_print(verbose, "*** compute_downsampled_images ***")

    images_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    images_nframes = len(images_filenames)
    images_zfill = len(images_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    image = myvtk.readImage(
        filename=images_folder+"/"+images_basename+"_"+str(0).zfill(images_zfill)+"."+images_ext,
        verbose=0)
    images_ndim = myvtk.getImageDimensionality(
        image=image,
        verbose=0)
    mypy.my_print(verbose, "images_ndim = "+str(images_ndim))
    images_dimensions = image.GetDimensions()
    mypy.my_print(verbose, "images_dimensions = "+str(images_dimensions))
    images_npoints = numpy.prod(images_dimensions)
    mypy.my_print(verbose, "images_npoints = "+str(images_npoints))
    images_origin = image.GetOrigin()
    mypy.my_print(verbose, "images_origin = "+str(images_origin))
    images_spacing = image.GetSpacing()
    mypy.my_print(verbose, "images_spacing = "+str(images_spacing))

    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))
    downsampling_factors = downsampling_factors+[1]*(3-images_ndim)
    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))

    images_downsampled_dimensions = numpy.divide(images_dimensions, downsampling_factors)
    images_downsampled_dimensions = numpy.ceil(images_downsampled_dimensions)
    images_downsampled_dimensions = [int(n) for n in images_downsampled_dimensions]
    mypy.my_print(verbose, "images_downsampled_dimensions = "+str(images_downsampled_dimensions))

    if   (images_ext == "vtk"):
        reader_type = vtk.vtkImageReader
        writer_type = vtk.vtkImageWriter
    elif (images_ext == "vti"):
        reader_type = vtk.vtkXMLImageDataReader
        writer_type = vtk.vtkXMLImageDataWriter
    else:
        assert 0, "\"ext\" must be \".vtk\" or \".vti\". Aborting."

    reader = reader_type()
    reader.UpdateDataObject()

    fft = vtk.vtkImageFFT()
    fft.SetDimensionality(images_ndim)
    fft.SetInputData(reader.GetOutput())
    fft.UpdateDataObject()
    if (write_temp_images):
        writer_fft = writer_type()
        writer_fft.SetInputData(fft.GetOutput())

    if (keep_resolution):
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
        for k_z in range(images_dimensions[2]):
            for k_y in range(images_dimensions[1]):
                for k_x in range(images_dimensions[0]):
                    k_point = k_z*images_dimensions[1]*images_dimensions[0] + k_y*images_dimensions[0] + k_x
                    if (((k_x >                        images_downsampled_dimensions[0]//2)  \
                    and  (k_x < images_dimensions[0] - images_downsampled_dimensions[0]//2)) \
                    or  ((k_y >                        images_downsampled_dimensions[1]//2)  \
                    and  (k_y < images_dimensions[1] - images_downsampled_dimensions[1]//2)) \
                    or  ((k_z >                        images_downsampled_dimensions[2]//2)  \
                    and  (k_z < images_dimensions[2] - images_downsampled_dimensions[2]//2))):
                        mask_scalars.SetTuple(k_point, [0, 0])
                    else:
                        mask_scalars.SetTuple(k_point, [1, 1])
        if (write_temp_images):
            myvtk.writeImage(
                image=mask_image,
                filename=images_folder+"/"+images_basename+"_mask"+"."+images_ext,
                verbose=0)

        mult = vtk.vtkImageMathematics()
        mult.SetOperationToMultiply()
        mult.SetInputData(0, fft.GetOutput())
        mult.SetInputData(1, mask_image)
        mult.UpdateDataObject()
        if (write_temp_images):
            writer_mul = writer_type()
            writer_mul.SetInputData(mult.GetOutput())
    else:
        images_downsampled_npoints = numpy.prod(images_downsampled_dimensions)
        mypy.my_print(verbose, "images_downsampled_npoints = "+str(images_downsampled_npoints))
        downsampling_factors = list(numpy.divide(images_dimensions, images_downsampled_dimensions))
        mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))
        downsampling_factor = numpy.prod(downsampling_factors)
        mypy.my_print(verbose, "downsampling_factor = "+str(downsampling_factor))
        images_downsampled_origin = images_origin
        mypy.my_print(verbose, "images_downsampled_origin = "+str(images_downsampled_origin))
        images_downsampled_spacing = list(numpy.multiply(images_spacing, downsampling_factors))
        mypy.my_print(verbose, "images_downsampled_spacing = "+str(images_downsampled_spacing))

        image_downsampled = vtk.vtkImageData()
        image_downsampled.SetDimensions(images_downsampled_dimensions)
        image_downsampled.SetOrigin(images_downsampled_origin)
        image_downsampled.SetSpacing(images_downsampled_spacing)

        image_downsampled_scalars = myvtk.createDoubleArray(
            name="ImageScalars",
            n_components=2,
            n_tuples=images_downsampled_npoints,
            verbose=0)
        image_downsampled.GetPointData().SetScalars(image_downsampled_scalars)
        I = numpy.empty(2)

        if (write_temp_images):
            writer_sel = writer_type()
            writer_sel.SetInputData(image_downsampled)

    rfft = vtk.vtkImageRFFT()
    rfft.SetDimensionality(images_ndim)
    if (keep_resolution):
        rfft.SetInputData(mult.GetOutput())
    else:
        rfft.SetInputData(image_downsampled)
    rfft.UpdateDataObject()

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputData(rfft.GetOutput())
    extract.SetComponents(0)
    extract.UpdateDataObject()

    writer = writer_type()
    writer.SetInputData(extract.GetOutput())

    for k_frame in range(images_nframes):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        reader.Update()

        fft.Update()
        if (write_temp_images):
            writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer_fft.Write()

        if (keep_resolution):

            mult.Update()
            if (write_temp_images):
                writer_mul.SetFileName(images_folder+"/"+images_basename+"_mul"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_mul.Write()

        else:

            image_scalars = fft.GetOutput().GetPointData().GetScalars()
            image_downsampled_scalars = image_downsampled.GetPointData().GetScalars()
            for k_z_downsampled in range(images_downsampled_dimensions[2]):
                k_z = k_z_downsampled if (k_z_downsampled <= images_downsampled_dimensions[2]//2) else k_z_downsampled+(images_dimensions[2]-images_downsampled_dimensions[2])
                for k_y_downsampled in range(images_downsampled_dimensions[1]):
                    k_y = k_y_downsampled if (k_y_downsampled <= images_downsampled_dimensions[1]//2) else k_y_downsampled+(images_dimensions[1]-images_downsampled_dimensions[1])
                    for k_x_downsampled in range(images_downsampled_dimensions[0]):
                        k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_dimensions[0]//2) else k_x_downsampled+(images_dimensions[0]-images_downsampled_dimensions[0])
                        k_point_downsampled = k_z_downsampled*images_downsampled_dimensions[1]*images_downsampled_dimensions[0] + k_y_downsampled*images_downsampled_dimensions[0] + k_x_downsampled
                        k_point             = k_z            *images_dimensions[1]            *images_dimensions[0]             + k_y            *images_dimensions[0]             + k_x
                        image_scalars.GetTuple(k_point, I)
                        I /= downsampling_factor
                        image_downsampled_scalars.SetTuple(k_point_downsampled, I)
            image_downsampled.Modified()

            if (write_temp_images):
                writer_sel.SetFileName(images_folder+"/"+images_basename+"_sel"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_sel.Write()

        rfft.Update()

        extract.Update()

        writer.SetFileName(images_folder+"/"+images_basename+("_"+suffix)*(suffix!="")+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        writer.Write()
