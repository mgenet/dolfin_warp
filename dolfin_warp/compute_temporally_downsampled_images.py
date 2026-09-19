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
import os
import vtk
import vtk.util.numpy_support

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_temporally_downsampled_images(
        images_folder                        ,
        images_basename                      ,
        temporal_downsampling_factor         ,
        temporal_window_size         = None  ,
        images_ext                   = "vti" ,
        suffix                       = None  ,
        verbose                      = 0     ):

    mypy.my_print(verbose, "*** compute_temporally_downsampled_images ***")

    image_series = dwarp.ImageSeries(
        folder   = images_folder   ,
        basename = images_basename ,
        ext      = images_ext      )

    n_frames = image_series.n_frames
    mypy.my_print(verbose, "n_frames = "+str(n_frames))

    assert (temporal_downsampling_factor >= 1),\
        "\"temporal_downsampling_factor\" (="+str(temporal_downsampling_factor)+") must not be smaller than one. Aborting."
    mypy.my_print(verbose, "temporal_downsampling_factor = "+str(temporal_downsampling_factor))

    # The frames are considered to have a finite temporal support, i.e., frame k covers [k-1/2, k+1/2], so that the series covers [-1/2, n_frames-1/2]. That same interval is then redistributed over n_frames_downsampled frames of equal width, so that no frame is ever dropped and all downsampled frames are averaged over the same duration. As for the spatial downsampling, the requested factor is thus only attained exactly when it divides the number of frames.
    n_frames_downsampled = math.ceil(n_frames/temporal_downsampling_factor)
    mypy.my_print(verbose, "n_frames_downsampled = "+str(n_frames_downsampled))

    effective_temporal_downsampling_factor = n_frames/n_frames_downsampled
    mypy.my_print(verbose, "effective_temporal_downsampling_factor = "+str(effective_temporal_downsampling_factor))

    if (temporal_window_size is None): # by default each downsampled frame is averaged over its own support
        temporal_window_size = effective_temporal_downsampling_factor
    assert (temporal_window_size > 0),\
        "\"temporal_window_size\" (="+str(temporal_window_size)+") must be positive. Aborting."
    mypy.my_print(verbose, "temporal_window_size = "+str(temporal_window_size))

    if   (images_ext == "vtk"):
        reader_type = vtk.vtkImageReader
        writer_type = vtk.vtkImageWriter
    elif (images_ext == "vti"):
        reader_type = vtk.vtkXMLImageDataReader
        writer_type = vtk.vtkXMLImageDataWriter
    else:
        assert 0, "\"images_ext\" (="+str(images_ext)+") must be \".vtk\" or \".vti\". Aborting."

    reader = reader_type()
    reader.SetFileName(image_series.get_image_filename(k_frame=0))
    reader.Update()

    image_downsampled = vtk.vtkImageData()
    image_downsampled.DeepCopy(reader.GetOutput())
    scalars_downsampled_vtk = image_downsampled.GetPointData().GetScalars()
    scalars_downsampled_np = vtk.util.numpy_support.vtk_to_numpy(scalars_downsampled_vtk)

    sum_scalars = numpy.empty(scalars_downsampled_np.shape) # the images may be of integer type, so they may not be summed in place

    writer = writer_type()
    writer.SetInputData(image_downsampled)

    if (suffix is None):
        suffix = "tdown="+str(temporal_downsampling_factor)

    for k_frame_downsampled in range(n_frames_downsampled):
        mypy.my_print(verbose, "k_frame_downsampled = "+str(k_frame_downsampled))

        k_frame_cen = -1/2 + (k_frame_downsampled + 1/2)*effective_temporal_downsampling_factor
        k_frame_ini = k_frame_cen - temporal_window_size/2
        k_frame_fin = k_frame_cen + temporal_window_size/2

        sum_scalars[:] = 0.
        sum_weights    = 0.
        for k_frame in range(max(math.ceil(k_frame_ini - 1/2), 0), min(math.floor(k_frame_fin + 1/2), n_frames-1)+1):
            weight = min(k_frame_fin, k_frame+1/2) - max(k_frame_ini, k_frame-1/2)
            if (weight <= 0.):
                continue
            mypy.my_print(verbose-1, "k_frame = "+str(k_frame)+" (weight = "+str(weight)+")")

            reader.SetFileName(image_series.get_image_filename(k_frame=k_frame))
            reader.Update()
            sum_scalars += weight * vtk.util.numpy_support.vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
            sum_weights += weight

        sum_scalars /= sum_weights
        scalars_downsampled_np[:] = sum_scalars

        scalars_downsampled_vtk.Modified()
        image_downsampled.Modified()

        writer.SetFileName(image_series.get_image_filename(k_frame=k_frame_downsampled, suffix=suffix))
        writer.Write()

if (__name__ == "__main__"):
    import fire
    fire.Fire(compute_temporally_downsampled_images)
