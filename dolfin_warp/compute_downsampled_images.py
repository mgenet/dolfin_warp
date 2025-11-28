#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2025                                       ###
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

def compute_downsampled_images(
        images_folder,
        images_basename,
        downsampling_factors,
        images_ext="vti",
        keep_resolution=0,
        write_temp_images=0,
        suffix=None,
        verbose=0):

    mypy.my_print(verbose, "*** compute_downsampled_images ***")

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

    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))
    downsampling_factors = downsampling_factors+[1]*(3-images_ndim)
    mypy.my_print(verbose, "downsampling_factors = "+str(downsampling_factors))

    images_downsampled_dimensions = numpy.ceil(numpy.divide(images_dimensions, downsampling_factors)).astype(int)
    mypy.my_print(verbose, "images_downsampled_dimensions = "+str(images_downsampled_dimensions))

    effective_downsampling_factors = numpy.divide(images_dimensions, images_downsampled_dimensions)
    print ("effective_downsampling_factors:", effective_downsampling_factors)

    effective_downsampling_factor = numpy.prod(effective_downsampling_factors)
    print ("effective_downsampling_factor:", effective_downsampling_factor)

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

    if (keep_resolution):
       image_downsampled_fft = image_fft
    else:
        images_downsampled_origin = images_origin
        mypy.my_print(verbose, "images_downsampled_origin = "+str(images_downsampled_origin))
        images_downsampled_spacing = list(numpy.multiply(images_spacing, effective_downsampling_factors))
        mypy.my_print(verbose, "images_downsampled_spacing = "+str(images_downsampled_spacing))

        image_downsampled_fft = myvtk.createImage(
            origin=images_downsampled_origin,
            spacing=images_downsampled_spacing,
            dimensions=images_downsampled_dimensions,
            array_name="ImageScalars",
            array_n_components=2)

    if (write_temp_images):
        writer_sel = writer_type()
        writer_sel.SetInputData(image_downsampled_fft)

    rfft_filter = vtk.vtkImageRFFT()
    rfft_filter.SetDimensionality(images_ndim)
    rfft_filter.SetInputData(image_downsampled_fft)
    rfft_filter.UpdateDataObject()

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputData(rfft_filter.GetOutput())
    extract.SetComponents(0)
    extract.UpdateDataObject()
    image_downsampled = extract.GetOutput()

    writer = writer_type()
    writer.SetInputData(image_downsampled)

    I = numpy.empty(2)
    for k_frame in range(image_series.n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(image_series.get_image_filename(k_frame=k_frame))
        reader.Update()

        fft_filter.Update()
        if (write_temp_images):
            writer_fft.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix="fft"))
            writer_fft.Write()

        if (keep_resolution):

            for k_z in range(images_dimensions[2]):
             for k_y in range(images_dimensions[1]):
              for k_x in range(images_dimensions[0]):
                if (((k_x >                        images_downsampled_dimensions[0]//2)  \
                and  (k_x < images_dimensions[0] - images_downsampled_dimensions[0]//2)) \
                or  ((k_y >                        images_downsampled_dimensions[1]//2)  \
                and  (k_y < images_dimensions[1] - images_downsampled_dimensions[1]//2)) \
                or  ((k_z >                        images_downsampled_dimensions[2]//2)  \
                and  (k_z < images_dimensions[2] - images_downsampled_dimensions[2]//2))):
                    image_downsampled_fft.SetScalarComponentFromDouble(k_x, k_y, k_z, 0, 0.)
                    image_downsampled_fft.SetScalarComponentFromDouble(k_x, k_y, k_z, 1, 0.)

        else:

            has_nyq_x = (images_downsampled_dimensions[0] % 2 == 0)
            has_nyq_y = (images_downsampled_dimensions[1] % 2 == 0)
            has_nyq_z = (images_downsampled_dimensions[2] % 2 == 0)

            for k_z in range(images_downsampled_dimensions[2]):
                is_nyq_z = has_nyq_z and (k_z == images_downsampled_dimensions[2] // 2)
                
                # 1. Identify Source Indices for Z
                base_k_z = k_z if (k_z <= images_downsampled_dimensions[2] // 2) else k_z + (images_dimensions[2] - images_downsampled_dimensions[2])
                alias_k_z = (images_dimensions[2] - base_k_z) % images_dimensions[2]
                
                # If Nyquist: Sum Base + Alias. Else: Just Base.
                source_indices_z = [base_k_z, alias_k_z] if is_nyq_z else [base_k_z]

                for k_y in range(images_downsampled_dimensions[1]):
                    is_nyq_y = has_nyq_y and (k_y == images_downsampled_dimensions[1] // 2)
                    
                    # 1. Identify Source Indices for Y
                    base_k_y = k_y if (k_y <= images_downsampled_dimensions[1] // 2) else k_y + (images_dimensions[1] - images_downsampled_dimensions[1])
                    alias_k_y = (images_dimensions[1] - base_k_y) % images_dimensions[1]
                    
                    # If Nyquist: Sum Base + Alias. Else: Just Base.
                    source_indices_y = [base_k_y, alias_k_y] if is_nyq_y else [base_k_y]

                    for k_x in range(images_downsampled_dimensions[0]):
                        is_nyq_x = has_nyq_x and (k_x == images_downsampled_dimensions[0] // 2)
                        
                        # 1. Identify Source Indices for Y
                        base_k_x = k_x if (k_x <= images_downsampled_dimensions[0] // 2) else k_x + (images_dimensions[0] - images_downsampled_dimensions[0])
                        alias_k_x = (images_dimensions[0] - base_k_x) % images_dimensions[0]
                        
                        # If Nyquist: Sum Base + Alias. Else: Just Base.
                        source_indices_x = [base_k_x, alias_k_x] if is_nyq_x else [base_k_x]

                        sum_r = 0.0
                        sum_i = 0.0

                        # === COMBINATORIAL SUMMATION ===
                        # Iterate directly over the explicit source indices
                        
                        for source_k_z in source_indices_z:
                            for source_k_y in source_indices_y:
                                for source_k_x in source_indices_x:

                                    # Accumulate Energy
                                    sum_r += image_fft.GetScalarComponentAsDouble(source_k_x, source_k_y, source_k_z, 0)
                                    sum_i += image_fft.GetScalarComponentAsDouble(source_k_x, source_k_y, source_k_z, 1)

                        # Scale
                        sum_r /= effective_downsampling_factor
                        sum_i /= effective_downsampling_factor

                        # Set Output
                        image_downsampled_fft.SetScalarComponentFromDouble(k_x, k_y, k_z, 0, sum_r)
                        image_downsampled_fft.SetScalarComponentFromDouble(k_x, k_y, k_z, 1, sum_i)

        image_downsampled_fft.Modified()

        if (write_temp_images):
            writer_sel.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix="sel"))
            writer_sel.Write()

        rfft_filter.Update()

        extract.Update()

        writer.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix=suffix))
        writer.Write()

        # image_scalars_np = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())
        # print("image_norm", numpy.linalg.norm(image_scalars_np))

        # image_fft_scalars_np = vtk.util.numpy_support.vtk_to_numpy(image_fft.GetPointData().GetScalars())
        # print("image_fft_norm", numpy.linalg.norm(image_fft_scalars_np)/math.sqrt(images_dimensions[2]*images_dimensions[1]*images_dimensions[0]))

        # image_downsampled_fft_scalars_np = vtk.util.numpy_support.vtk_to_numpy(image_downsampled_fft.GetPointData().GetScalars())
        # print("image_downsampled_fft_norm", numpy.linalg.norm(image_downsampled_fft_scalars_np)/math.sqrt(images_downsampled_dimensions[2]*images_downsampled_dimensions[1]*images_downsampled_dimensions[0]))

        # image_downsampled_scalars_np = vtk.util.numpy_support.vtk_to_numpy(image_downsampled.GetPointData().GetScalars())
        # print("image_downsampled_norm", numpy.linalg.norm(image_downsampled_scalars_np))
