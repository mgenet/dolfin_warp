#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2020                                       ###
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

import dolfin_dic as ddic

################################################################################

def compute_downsampled_images(
        images_folder,
        images_basename,
        downsampling_factors,
        images_ext="vti",
        keep_resolution=0,
        overwrite_orig_images=1,
        write_temp_images=0,
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
    images_spacing = image.GetSpacing()
    mypy.my_print(verbose, "images_spacing = "+str(images_spacing))
    images_delta = images_spacing[0:images_ndim]
    mypy.my_print(verbose, "images_delta = "+str(images_delta))
    images_nvoxels = myvtk.getImageDimensions(
        image=image,
        verbose=0)
    mypy.my_print(verbose, "images_nvoxels = "+str(images_nvoxels))
    images_npoints = numpy.prod(images_nvoxels)
    mypy.my_print(verbose, "images_npoints = "+str(images_npoints))

    images_downsampled_nvoxels = numpy.divide(images_nvoxels, downsampling_factors)
    images_downsampled_nvoxels = numpy.ceil(images_downsampled_nvoxels)
    images_downsampled_nvoxels = [int(n) for n in images_downsampled_nvoxels]
    mypy.my_print(verbose, "images_downsampled_nvoxels = "+str(images_downsampled_nvoxels))
    downsampling_factors = list(numpy.divide(images_nvoxels, images_downsampled_nvoxels))
    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))
    downsampling_factor = numpy.prod(downsampling_factors)
    mypy.my_print(verbose, "downsampling_factor = "+str(downsampling_factor))
    images_downsampled_delta = list(numpy.multiply(images_delta, downsampling_factors))
    # mypy.my_print(verbose, "images_downsampled_delta = "+str(images_downsampled_delta))
    images_downsampled_npoints = numpy.prod(images_downsampled_nvoxels)
    # mypy.my_print(verbose, "images_downsampled_npoints = "+str(images_downsampled_npoints))

    if   (images_ext == "vtk"):
        reader_constr = vtk.vtkImageReader
        writer_constr = vtk.vtkImageWriter
    elif (images_ext == "vti"):
        reader_constr = vtk.vtkXMLImageDataReader
        writer_constr = vtk.vtkXMLImageDataWriter
    else:
        assert 0, "\"ext\" must be \".vtk\" or \".vti\". Aborting."
    reader = reader_constr()
    writer = writer_constr()
    if (write_temp_images):
        writer_fft  = writer_constr()
        if (keep_resolution):
            writer_mul = writer_constr()
        else:
            writer_sel = writer_constr()

    fft = vtk.vtkImageFFT()
    fft.SetDimensionality(images_ndim)
    fft.SetInputConnection(reader.GetOutputPort())
    if (write_temp_images):
        writer_fft.SetInputConnection(fft.GetOutputPort())

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
        # print(mask_image.GetScalarType())
        # print(mask_image.GetPointData().GetScalars())
        if (images_ndim == 1):
            for k_x in range(images_nvoxels[0]):
                if ((k_x >                     images_downsampled_nvoxels[0]//2) \
                and (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]//2)):
                    mask_scalars.SetTuple(k_x, [0, 0])
                else:
                    mask_scalars.SetTuple(k_x, [1, 1])
        if (images_ndim == 2):
            for k_y in range(images_nvoxels[1]):
                for k_x in range(images_nvoxels[0]):
                    k_point = k_y*images_nvoxels[0] + k_x
                    if (((k_x >                     images_downsampled_nvoxels[0]//2)  \
                    and  (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]//2)) \
                    or  ((k_y >                     images_downsampled_nvoxels[1]//2)  \
                    and  (k_y < images_nvoxels[1] - images_downsampled_nvoxels[1]//2))):
                        mask_scalars.SetTuple(k_point, [0, 0])
                    else:
                        mask_scalars.SetTuple(k_point, [1, 1])
        if (images_ndim == 3):
            for k_z in range(images_nvoxels[2]):
                for k_y in range(images_nvoxels[1]):
                    for k_x in range(images_nvoxels[0]):
                        k_point = k_z*images_nvoxels[1]*images_nvoxels[0] + k_y*images_nvoxels[0] + k_x
                        if (((k_x >                     images_downsampled_nvoxels[0]//2)  \
                        and  (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]//2)) \
                        or  ((k_y >                     images_downsampled_nvoxels[1]//2)  \
                        and  (k_y < images_nvoxels[1] - images_downsampled_nvoxels[1]//2)) \
                        or  ((k_z >                     images_downsampled_nvoxels[2]//2)  \
                        and  (k_z < images_nvoxels[2] - images_downsampled_nvoxels[2]//2))):
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
        mult.SetInputConnection(0, fft.GetOutputPort())
        mult.SetInputData(1, mask_image)
        if (write_temp_images):
            writer_mul.SetInputConnection(mult.GetOutputPort())
    else:
        image_downsampled = vtk.vtkImageData()

        dimensions_downsampled = images_downsampled_nvoxels+[1]*(3-images_ndim)
        mypy.my_print(verbose, "dimensions_downsampled = "+str(dimensions_downsampled))
        image_downsampled.SetDimensions(dimensions_downsampled)

        spacing_downsampled = images_downsampled_delta+[1.]*(3-images_ndim)
        mypy.my_print(verbose, "spacing_downsampled = "+str(spacing_downsampled))
        image_downsampled.SetSpacing(spacing_downsampled)

        origin_downsampled = list(numpy.divide(images_downsampled_delta, 2))
        origin_downsampled = origin_downsampled+[0.]*(3-images_ndim)
        mypy.my_print(verbose, "origin_downsampled = "+str(origin_downsampled))
        image_downsampled.SetOrigin(origin_downsampled)

        image_downsampled_scalars = myvtk.createDoubleArray(
            name="ImageScalars",
            n_components=2,
            n_tuples=images_downsampled_npoints,
            verbose=0)
        image_downsampled.GetPointData().SetScalars(image_downsampled_scalars)
        I = numpy.empty(2)

        if (write_temp_images):
            writer_sel.SetInputData(image_downsampled)

    rfft = vtk.vtkImageRFFT()
    rfft.SetDimensionality(images_ndim)
    if (keep_resolution):
        rfft.SetInputConnection(mult.GetOutputPort())
    else:
        rfft.SetInputData(image_downsampled) # MG20190520: Not sure why this does not work.

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputConnection(rfft.GetOutputPort())
    extract.SetComponents(0)

    writer.SetInputConnection(extract.GetOutputPort())

    if (keep_resolution):
        for k_frame in range(images_nframes):
            mypy.my_print(verbose, "k_frame = "+str(k_frame))

            reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)

            if (write_temp_images):
                writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_fft.Write()

            if (write_temp_images):
                writer_mul.SetFileName(images_folder+"/"+images_basename+"_mul"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_mul.Write()

            writer.SetFileName(images_folder+"/"+images_basename+("_downsampled")*(not overwrite_orig_images)+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer.Write()
    else:
        for k_frame in range(images_nframes):
            mypy.my_print(verbose, "k_frame = "+str(k_frame))

            reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            reader.Update()
            # print("reader.GetOutput() = "+str(reader.GetOutput()))

            fft.Update()
            if (write_temp_images):
                writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_fft.Write()
            # print("fft.GetOutput() = "+str(fft.GetOutput()))
            # print("fft.GetOutput().GetScalarType() = "+str(fft.GetOutput().GetScalarType()))
            # print("fft.GetOutput().GetPointData().GetScalars() = "+str(fft.GetOutput().GetPointData().GetScalars()))

            image_scalars = fft.GetOutput().GetPointData().GetScalars()
            if (images_ndim == 1):
                for k_x_downsampled in range(images_downsampled_nvoxels[0]):
                    k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]//2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                    image_scalars.GetTuple(k_x, I)
                    I /= downsampling_factor
                    image_downsampled_scalars.SetTuple(k_x_downsampled, I)
            if (images_ndim == 2):
                for k_y_downsampled in range(images_downsampled_nvoxels[1]):
                    k_y = k_y_downsampled if (k_y_downsampled <= images_downsampled_nvoxels[1]//2) else k_y_downsampled+(images_nvoxels[1]-images_downsampled_nvoxels[1])
                    for k_x_downsampled in range(images_downsampled_nvoxels[0]):
                        k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]//2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                        k_point_downsampled = k_y_downsampled*images_downsampled_nvoxels[0] + k_x_downsampled
                        k_point             = k_y            *images_nvoxels[0]             + k_x
                        image_scalars.GetTuple(k_point, I)
                        I /= downsampling_factor
                        image_downsampled_scalars.SetTuple(k_point_downsampled, I)
            if (images_ndim == 3):
                for k_z_downsampled in range(images_downsampled_nvoxels[2]):
                    k_z = k_z_downsampled if (k_z_downsampled <= images_downsampled_nvoxels[2]//2) else k_z_downsampled+(images_nvoxels[2]-images_downsampled_nvoxels[2])
                    for k_y_downsampled in range(images_downsampled_nvoxels[1]):
                        k_y = k_y_downsampled if (k_y_downsampled <= images_downsampled_nvoxels[1]//2) else k_y_downsampled+(images_nvoxels[1]-images_downsampled_nvoxels[1])
                        for k_x_downsampled in range(images_downsampled_nvoxels[0]):
                            k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]//2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                            k_point_downsampled = k_z_downsampled*images_downsampled_nvoxels[1]*images_downsampled_nvoxels[0] + k_y_downsampled*images_downsampled_nvoxels[0] + k_x_downsampled
                            k_point             = k_z            *images_nvoxels[1]            *images_nvoxels[0]             + k_y            *images_nvoxels[0]             + k_x
                            image_scalars.GetTuple(k_point, I)
                            I /= downsampling_factor
                            image_downsampled_scalars.SetTuple(k_point_downsampled, I)
            # print("image_downsampled = "+str(image_downsampled))
            # print("image_downsampled_scalars = "+str(image_downsampled_scalars))

            if (write_temp_images):
                writer_sel.SetFileName(images_folder+"/"+images_basename+"_sel"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_sel.Write()

            rfft = vtk.vtkImageRFFT()                 # MG20190520: Not sure why this is needed.
            rfft.SetDimensionality(images_ndim)       # MG20190520: Not sure why this is needed.
            rfft.SetInputData(image_downsampled)      # MG20190520: Not sure why this is needed.
            rfft.Update()

            extract = vtk.vtkImageExtractComponents() # MG20190520: Not sure why this is needed.
            extract.SetInputData(rfft.GetOutput())    # MG20190520: Not sure why this is needed.
            extract.SetComponents(0)                  # MG20190520: Not sure why this is needed.
            extract.Update()                          # MG20190520: Not sure why this is needed.

            writer.SetInputData(extract.GetOutput())  # MG20190520: Not sure why this is needed.
            writer.SetFileName(images_folder+"/"+images_basename+("_downsampled")*(not overwrite_orig_images)+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer.Write()
