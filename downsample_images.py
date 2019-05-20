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

def downsample_images(
        images_folder,
        images_basename,
        downsampling_factors,
        images_ext="vti",
        keep_resolution=0,
        write_temp_images=0,
        verbose=0):

    mypy.my_print(verbose, "*** downsample_images ***")

    images_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    images_nframes = len(images_filenames)
    images_zfill = len(images_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    image = myvtk.readImage(
        filename=images_folder+"/"+images_basename+"_"+str(0).zfill(images_zfill)+"."+images_ext,
        verbose=0)
    images_ndim = myvtk.getImageDimensionality(
        image=image,
        verbose=0)
    images_spacing = image.GetSpacing()
    images_origin = image.GetOrigin()
    images_nvoxels = myvtk.getImageDimensions(
        image=image,
        verbose=0)
    mypy.my_print(verbose, "images_nvoxels = "+str(images_nvoxels))
    assert (len(images_nvoxels) == images_ndim)
    images_npoints = numpy.prod(images_nvoxels)
    # mypy.my_print(verbose, "images_npoints = "+str(images_npoints))
    assert (images_npoints == image.GetNumberOfPoints())

    images_downsampled_nvoxels = numpy.array(images_nvoxels)/numpy.array(downsampling_factors)
    images_downsampled_nvoxels = numpy.ceil(images_downsampled_nvoxels)
    images_downsampled_nvoxels = [int(n) for n in images_downsampled_nvoxels]
    mypy.my_print(verbose, "images_downsampled_nvoxels = "+str(images_downsampled_nvoxels))
    downsampling_factors = numpy.array(images_nvoxels)/numpy.array(images_downsampled_nvoxels)
    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))
    downsampling_factor = numpy.prod(downsampling_factors)
    mypy.my_print(verbose, "downsampling_factor = "+str(downsampling_factor))
    images_downsampled_npoints = numpy.prod(images_downsampled_nvoxels)
    # mypy.my_print(verbose, "images_downsampled_npoints = "+str(images_downsampled_npoints))

    if   (images_ext == "vtk"):
        reader = vtk.vtkImageReader()
        writer = vtk.vtkImageWriter()
        if (write_temp_images):
            writer_fft  = vtk.vtkImageWriter()
            if (keep_resolution):
                writer_mul = vtk.vtkImageWriter()
            else:
                writer_sel = vtk.vtkImageWriter()
    elif (images_ext == "vti"):
        reader = vtk.vtkXMLImageDataReader()
        writer = vtk.vtkXMLImageDataWriter()
        if (write_temp_images):
            writer_fft  = vtk.vtkXMLImageDataWriter()
            if (keep_resolution):
                writer_mul = vtk.vtkXMLImageDataWriter()
            else:
                writer_sel = vtk.vtkXMLImageDataWriter()
    else:
        assert 0, "\"ext\" must be \".vtk\" or \".vti\". Aborting."

    fft = vtk.vtkImageFFT()
    fft.SetDimensionality(images_ndim)
    fft.SetInputConnection(reader.GetOutputPort())
    if (write_temp_images): writer_fft.SetInputConnection(fft.GetOutputPort())

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
        # print mask_image.GetScalarType()
        # print mask_image.GetPointData().GetScalars()
        if (images_ndim == 1):
            for k_x in xrange(images_nvoxels[0]):
                if ((k_x >                     images_downsampled_nvoxels[0]/2) \
                and (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]/2)):
                    mask_scalars.SetTuple(k_x, [0, 0])
                else:
                    mask_scalars.SetTuple(k_x, [1, 1])
        if (images_ndim == 2):
            for k_y in xrange(images_nvoxels[1]):
                for k_x in xrange(images_nvoxels[0]):
                    k_point = k_y*images_nvoxels[0] + k_x
                    if (((k_x >                     images_downsampled_nvoxels[0]/2)  \
                    and  (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]/2)) \
                    or  ((k_y >                     images_downsampled_nvoxels[1]/2)  \
                    and  (k_y < images_nvoxels[1] - images_downsampled_nvoxels[1]/2))):
                        mask_scalars.SetTuple(k_point, [0, 0])
                    else:
                        mask_scalars.SetTuple(k_point, [1, 1])
        if (images_ndim == 3):
            for k_z in xrange(images_nvoxels[2]):
                for k_y in xrange(images_nvoxels[1]):
                    for k_x in xrange(images_nvoxels[0]):
                        k_point = k_z*images_nvoxels[1]*images_nvoxels[0] + k_y*images_nvoxels[0] + k_x
                        if (((k_x >                     images_downsampled_nvoxels[0]/2)  \
                        and  (k_x < images_nvoxels[0] - images_downsampled_nvoxels[0]/2)) \
                        or  ((k_y >                     images_downsampled_nvoxels[1]/2)  \
                        and  (k_y < images_nvoxels[1] - images_downsampled_nvoxels[1]/2)) \
                        or  ((k_z >                     images_downsampled_nvoxels[2]/2)  \
                        and  (k_z < images_nvoxels[2] - images_downsampled_nvoxels[2]/2))):
                            mask_scalars.SetTuple(k_point, [0, 0])
                        else:
                            mask_scalars.SetTuple(k_point, [1, 1])
        if (write_temp_images): myvtk.writeImage(
            image=mask_image,
            filename=images_folder+"/"+images_basename+"_mask"+"."+images_ext,
            verbose=0)

        mult = vtk.vtkImageMathematics()
        mult.SetOperationToMultiply()
        mult.SetInputConnection(0, fft.GetOutputPort())
        mult.SetInputData(1, mask_image)
        if (write_temp_images): writer_mul.SetInputConnection(mult.GetOutputPort())
    else:
        image_downsampled = vtk.vtkImageData()

        if   (images_ndim == 1):
            extent = [0, images_downsampled_nvoxels[0]-1, 0,                               0, 0,                               0]
        elif (images_ndim == 2):
            extent = [0, images_downsampled_nvoxels[0]-1, 0, images_downsampled_nvoxels[1]-1, 0,                               0]
        elif (images_ndim == 3):
            extent = [0, images_downsampled_nvoxels[0]-1, 0, images_downsampled_nvoxels[1]-1, 0, images_downsampled_nvoxels[2]-1]
        image_downsampled.SetExtent(extent)
        mypy.my_print(verbose, "extent = "+str(extent))

        if   (images_ndim == 1):
            spacing = [images_spacing[0]*downsampling_factors[0],                                        1.,                                        1.]
        elif (images_ndim == 2):
            spacing = [images_spacing[0]*downsampling_factors[0], images_spacing[1]*downsampling_factors[1],                                        1.]
        elif (images_ndim == 3):
            spacing = [images_spacing[0]*downsampling_factors[0], images_spacing[1]*downsampling_factors[1], images_spacing[2]*downsampling_factors[2]]
        image_downsampled.SetSpacing(spacing)
        mypy.my_print(verbose, "spacing = "+str(spacing))

        if   (images_ndim == 1):
            origin = [spacing[0]/2,           0.,           0.]
        elif (images_ndim == 2):
            origin = [spacing[0]/2, spacing[1]/2,           0.]
        elif (images_ndim == 3):
            origin = [spacing[0]/2, spacing[1]/2, spacing[2]/2]
        image_downsampled.SetOrigin(origin)
        mypy.my_print(verbose, "origin = "+str(origin))

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

    writer.SetInputConnection(rfft.GetOutputPort())

    if (keep_resolution):
        for k_frame in xrange(images_nframes):
            mypy.my_print(verbose, "k_frame = "+str(k_frame))

            reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)

            if (write_temp_images):
                writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_fft.Write()

            if (write_temp_images):
                writer_mul.SetFileName(images_folder+"/"+images_basename+"_mul"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_mul.Write()

            writer.SetFileName(images_folder+"/"+images_basename+"_downsampled"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer.Write()
    else:
        for k_frame in xrange(images_nframes):
            mypy.my_print(verbose, "k_frame = "+str(k_frame))

            reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            reader.Update()
            # print "reader.GetOutput() = "+str(reader.GetOutput())

            fft.Update()
            if (write_temp_images):
                writer_fft.SetFileName(images_folder+"/"+images_basename+"_fft"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_fft.Write()
            # print "fft.GetOutput() = "+str(fft.GetOutput())
            # print "fft.GetOutput().GetScalarType() = "+str(fft.GetOutput().GetScalarType())
            # print "fft.GetOutput().GetPointData().GetScalars() = "+str(fft.GetOutput().GetPointData().GetScalars())

            image_scalars = fft.GetOutput().GetPointData().GetScalars()
            if (images_ndim == 1):
                for k_x_downsampled in xrange(images_downsampled_nvoxels[0]):
                    k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]/2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                    image_scalars.GetTuple(k_x, I)
                    I /= downsampling_factor
                    image_downsampled_scalars.SetTuple(k_x_downsampled, I)
            if (images_ndim == 2):
                for k_y_downsampled in xrange(images_downsampled_nvoxels[1]):
                    k_y = k_y_downsampled if (k_y_downsampled <= images_downsampled_nvoxels[1]/2) else k_y_downsampled+(images_nvoxels[1]-images_downsampled_nvoxels[1])
                    for k_x_downsampled in xrange(images_downsampled_nvoxels[0]):
                        k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]/2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                        k_point_downsampled = k_y_downsampled*images_downsampled_nvoxels[0] + k_x_downsampled
                        k_point             = k_y            *images_nvoxels[0]             + k_x
                        image_scalars.GetTuple(k_point, I)
                        I /= downsampling_factor
                        image_downsampled_scalars.SetTuple(k_point_downsampled, I)
            if (images_ndim == 3):
                for k_z_downsampled in xrange(images_downsampled_nvoxels[2]):
                    k_z = k_z_downsampled if (k_z_downsampled <= images_downsampled_nvoxels[2]/2) else k_z_downsampled+(images_nvoxels[2]-images_downsampled_nvoxels[2])
                    for k_y_downsampled in xrange(images_downsampled_nvoxels[1]):
                        k_y = k_y_downsampled if (k_y_downsampled <= images_downsampled_nvoxels[1]/2) else k_y_downsampled+(images_nvoxels[1]-images_downsampled_nvoxels[1])
                        for k_x_downsampled in xrange(images_downsampled_nvoxels[0]):
                            k_x = k_x_downsampled if (k_x_downsampled <= images_downsampled_nvoxels[0]/2) else k_x_downsampled+(images_nvoxels[0]-images_downsampled_nvoxels[0])
                            k_point_downsampled = k_z_downsampled*images_downsampled_nvoxels[1]*images_downsampled_nvoxels[0] + k_y_downsampled*images_downsampled_nvoxels[0] + k_x_downsampled
                            k_point             = k_z            *images_nvoxels[1]            *images_nvoxels[0]             + k_y            *images_nvoxels[0]             + k_x
                            image_scalars.GetTuple(k_point, I)
                            I /= downsampling_factor
                            image_downsampled_scalars.SetTuple(k_point_downsampled, I)
            # print "image_downsampled = "+str(image_downsampled)
            # print "image_downsampled_scalars = "+str(image_downsampled_scalars)

            if (write_temp_images):
                # writer_sel.SetInputData(image_downsampled)
                writer_sel.SetFileName(images_folder+"/"+images_basename+"_sel"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
                writer_sel.Write()

            rfft = vtk.vtkImageRFFT()             # MG20190520: Not sure why this is needed.
            rfft.SetDimensionality(images_ndim)   # MG20190520: Not sure why this is needed.
            rfft.SetInputData(image_downsampled)  # MG20190520: Not sure why this is needed.
            rfft.Update()

            writer.SetInputData(rfft.GetOutput()) # MG20190520: Not sure why this is needed.
            writer.SetFileName(images_folder+"/"+images_basename+"_downsampled"+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
            writer.Write()
