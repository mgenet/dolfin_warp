#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import math
import numpy
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_upsampled_images(
        images_folder              ,
        images_basename            ,
        upsampling_factors         ,
        images_ext         = "vti" ,
        write_temp_images  = 0     ,
        suffix             = None  ,
        verbose            = 0     ):

    mypy.my_print(verbose, "*** compute_upsampled_images ***")

    image_series = dwarp.ImageSeries(
        folder=images_folder,
        basename=images_basename,
        ext=images_ext)

    image = image_series.get_image(k_frame=0)
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

    mypy.my_print(verbose, "upsampling_factors = "+str(upsampling_factors))
    upsampling_factors = upsampling_factors+[1]*(3-images_ndim)
    mypy.my_print(verbose, "upsampling_factors = "+str(upsampling_factors))

    images_upsampled_dimensions = numpy.round(numpy.multiply(images_dimensions, upsampling_factors)).astype(int)
    mypy.my_print(verbose, "images_upsampled_dimensions = "+str(images_upsampled_dimensions))

    effective_upsampling_factors = numpy.divide(images_upsampled_dimensions, images_dimensions)
    print ("effective_upsampling_factors:", effective_upsampling_factors)

    effective_upsampling_factor = numpy.prod(effective_upsampling_factors)
    print ("effective_upsampling_factor:", effective_upsampling_factor)

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
    image = reader.GetOutput()

    fft_filter = vtk.vtkImageFFT()
    fft_filter.SetDimensionality(images_ndim)
    fft_filter.SetInputData(image)
    fft_filter.UpdateDataObject()
    image_fft = fft_filter.GetOutput()

    if (write_temp_images):
        writer_fft = writer_type()
        writer_fft.SetInputData(image_fft)

    images_upsampled_origin = images_origin
    mypy.my_print(verbose, "images_upsampled_origin = "+str(images_upsampled_origin))
    images_upsampled_spacing = list(numpy.divide(images_spacing, effective_upsampling_factors))
    mypy.my_print(verbose, "images_upsampled_spacing = "+str(images_upsampled_spacing))

    image_upsampled_fft = myvtk.createImage(
        origin=images_upsampled_origin,
        spacing=images_upsampled_spacing,
        dimensions=images_upsampled_dimensions,
        array_name="ImageScalars",
        array_n_components=2)

    if (write_temp_images):
        writer_sel = writer_type()
        writer_sel.SetInputData(image_upsampled_fft)

    rfft_filter = vtk.vtkImageRFFT()
    rfft_filter.SetDimensionality(images_ndim)
    rfft_filter.SetInputData(image_upsampled_fft)
    rfft_filter.UpdateDataObject()

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputData(rfft_filter.GetOutput())
    extract.SetComponents(0)
    extract.UpdateDataObject()
    image_upsampled = extract.GetOutput()

    writer = writer_type()
    writer.SetInputData(image_upsampled)

    for k_frame in range(image_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(image_series.get_image_filename(k_frame=k_frame))
        reader.Update()

        fft_filter.Update()
        if (write_temp_images):
            writer_fft.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix="fft"))
            writer_fft.Write()

        scalars = image_upsampled_fft.GetPointData().GetScalars()
        for k_component in range(scalars.GetNumberOfComponents()):
            scalars.FillComponent(k_component, 0.0)

        has_nyq_x = (images_dimensions[0] % 2 == 0)
        has_nyq_y = (images_dimensions[1] % 2 == 0)
        has_nyq_z = (images_dimensions[2] % 2 == 0)

        for k_z in range(images_dimensions[2]):
            is_nyq_z = has_nyq_z and (k_z == images_dimensions[2] // 2)
            target_k_z = k_z if (k_z <= images_dimensions[2] // 2) else k_z + (images_upsampled_dimensions[2] - images_dimensions[2])
            alias_k_z  = (images_upsampled_dimensions[2] - target_k_z) % images_upsampled_dimensions[2]
            target_indices_z = [target_k_z, alias_k_z] if (is_nyq_z and alias_k_z != target_k_z) else [target_k_z]

            for k_y in range(images_dimensions[1]):
                is_nyq_y = has_nyq_y and (k_y == images_dimensions[1] // 2)
                target_k_y = k_y if (k_y <= images_dimensions[1] // 2) else k_y + (images_upsampled_dimensions[1] - images_dimensions[1])
                alias_k_y  = (images_upsampled_dimensions[1] - target_k_y) % images_upsampled_dimensions[1]
                target_indices_y = [target_k_y, alias_k_y] if (is_nyq_y and alias_k_y != target_k_y) else [target_k_y]

                for k_x in range(images_dimensions[0]):
                    is_nyq_x = has_nyq_x and (k_x == images_dimensions[0] // 2)
                    target_k_x = k_x if (k_x <= images_dimensions[0] // 2) else k_x + (images_upsampled_dimensions[0] - images_dimensions[0])
                    alias_k_x  = (images_upsampled_dimensions[0] - target_k_x) % images_upsampled_dimensions[0]
                    target_indices_x = [target_k_x, alias_k_x] if (is_nyq_x and alias_k_x != target_k_x) else [target_k_x]

                    # Get Coarse Value
                    val_r = image_fft.GetScalarComponentAsDouble(k_x, k_y, k_z, 0)
                    val_i = image_fft.GetScalarComponentAsDouble(k_x, k_y, k_z, 1)

                    # Scale
                    val_r *= effective_upsampling_factor
                    val_i *= effective_upsampling_factor

                    # Hermitian Symmetry Split: If this is a Nyquist frequency, its energy must be split evenly between the positive and negative frequencies in the new padded grid.
                    num_targets = len(target_indices_z) * len(target_indices_y) * len(target_indices_x)
                    val_r /= num_targets
                    val_i /= num_targets

                    # Scatter to Fine Grid
                    for t_z in target_indices_z:
                     for t_y in target_indices_y:
                      for t_x in target_indices_x:
                        image_upsampled_fft.SetScalarComponentFromDouble(t_x, t_y, t_z, 0, val_r)
                        image_upsampled_fft.SetScalarComponentFromDouble(t_x, t_y, t_z, 1, val_i)

        image_upsampled_fft.Modified()

        if (write_temp_images):
            writer_sel.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix="sel"))
            writer_sel.Write()

        rfft_filter.Update()

        extract.Update()

        writer.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix=suffix))
        writer.Write()

if (__name__ == "__main__"):
    import fire
    fire.Fire(compute_upsampled_images)
