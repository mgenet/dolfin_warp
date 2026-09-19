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

def temporally_downsample_stacks(
        images                             ,
        directions                         ,
        temporal_downsampling_factor       ,
        temporal_window_size         = None,
        verbose                      = 0   ):

    # Each stack is temporally downsampled on its own, before the stacks are combined, as this is the order in which this happens during the acquisition (and as the combination, which involves a geometric mean, does not commute with the temporal averaging).
    # Returns the suffix under which the downsampled stacks have been written, and updates images["n_frames"] accordingly.

    if (temporal_downsampling_factor == 1) and (temporal_window_size is None):
        return ""

    suffix = "tdown="+str(temporal_downsampling_factor)

    for direction in directions:
        dwarp.compute_temporally_downsampled_images(
            images_folder                = images["folder"]                 ,
            images_basename              = images["basename"]+"-"+direction ,
            temporal_downsampling_factor = temporal_downsampling_factor     ,
            temporal_window_size         = temporal_window_size             ,
            images_ext                   = images["ext"]                    ,
            suffix                       = suffix                           ,
            verbose                      = verbose-1                        )

    images["n_frames"] = math.ceil(images["n_frames"]/temporal_downsampling_factor)
    mypy.my_print(verbose, "n_frames = "+str(images["n_frames"]))

    return "-"+suffix

################################################################################

def remove_stacks(
        image_series       ,
        directions         ,
        stack_suffix       ,
        n_frames_generated ,
        n_frames           ):

    for direction in directions:
        for k_frame in range(n_frames_generated):
            os.remove(image_series.get_image_filename(k_frame=k_frame, suffix=direction))
        if bool(stack_suffix):
            for k_frame in range(n_frames):
                os.remove(image_series.get_image_filename(k_frame=k_frame, suffix=direction+stack_suffix))

################################################################################

def get_undersampling_factors(
        images              ,
        k_dim               ,
        undersampling_level ):

    # Each stack is undersampled in all the directions but its own tagging direction.

    return [1 if (k == k_dim) else undersampling_level for k in range(images["n_dim"])]

################################################################################

def normalize_stacks(
        images           ,
        directions       ,
        stack_suffix     ,
        verbose      = 0 ):

    # Each stack is mapped linearly onto [0, 1], globally over all the frames of the stack, so that the mapping does not vary in time.
    # Note that this lifts the background off zero, by as much as the largest undershoot of the stack.

    for direction in directions:
        dwarp.compute_normalized_images(
            images_folder   = images["folder"]                            ,
            images_basename = images["basename"]+"-"+direction+stack_suffix,
            images_datatype = "ufloat"                                    ,
            images_ext      = images["ext"]                               ,
            verbose         = verbose-1                                   )

################################################################################

def generate_undersampled_images_through_linear_interpolation(
        images                           ,
        structure                        ,
        texture                          ,
        noise                            ,
        deformation                      ,
        evolution                        ,
        undersampling_level              ,
        stacks_normalization  = "linear" , # "linear" or "clip"
        keep_temporary_images = 0        ,
        verbose               = 0        ):

    mypy.my_print(verbose, "*** generate_undersampled_images_through_linear_interpolation ***")

    assert (stacks_normalization in ("linear", "clip")),\
        "\"stacks_normalization\" (=\""+str(stacks_normalization)+"\") must be \"clip\" or \"linear\". Aborting."

    images_basename = images["basename"]
    images_n_voxels = images["n_voxels"][:]
    images_n_frames = images["n_frames"]
    texture_type    = texture["type"]

    temporal_downsampling_factor = images["temporal_downsampling_factor"] if ("temporal_downsampling_factor" in images) else 1
    temporal_window_size         = images["temporal_window_size"]         if ("temporal_window_size"         in images) else None
    images["temporal_downsampling_factor"] = 1 # the temporal downsampling is applied to each stack below, once it has been generated
    images["temporal_window_size"]         = None

    directions = ["X", "Y", "Z"][:images["n_dim"]]

    # Each stack is generated on its own grid, coarse in the directions orthogonal to its tagging direction.
    for k_dim, direction in enumerate(directions):
        images["n_voxels"][:] = images_n_voxels[:]
        for k in range(images["n_dim"]):
            if (k != k_dim):
                images["n_voxels"][k] //= undersampling_level
        images["basename"] = images_basename+"-"+direction
        images["n_frames"] = images_n_frames
        texture["type"]    = "tagg"+direction

        dwarp.generate_images(
            images      = images      ,
            structure   = structure   ,
            texture     = texture     ,
            noise       = noise       ,
            deformation = deformation ,
            evolution   = evolution   ,
            verbose     = verbose-1   )

    images["n_voxels"][:]                  = images_n_voxels[:]
    images["basename"]                     = images_basename
    images["n_frames"]                     = images_n_frames
    images["temporal_downsampling_factor"] = temporal_downsampling_factor
    images["temporal_window_size"]         = temporal_window_size
    texture["type"]                        = texture_type

    stack_suffix = temporally_downsample_stacks(
        images                       = images                       ,
        directions                   = directions                   ,
        temporal_downsampling_factor = temporal_downsampling_factor ,
        temporal_window_size         = temporal_window_size         ,
        verbose                      = verbose                      )

    if (stacks_normalization == "linear"):
        normalize_stacks(
            images       = images       ,
            directions   = directions   ,
            stack_suffix = stack_suffix ,
            verbose      = verbose      )

    image_series = dwarp.ImageSeries( # the combined images do not exist yet, so the series is declared, not read
        folder   = images["folder"]   ,
        basename = images["basename"] ,
        n_frames = images["n_frames"] ,
        zfill    = images["zfill"]    ,
        ext      = images["ext"]      ,
        verbose  = 0                  )

    x = numpy.empty(3)
    i = numpy.empty(1)
    for k_frame in range(images["n_frames"]):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        interpolators = [myvtk.getImageInterpolator(
            image   = myvtk.readImage(
                filename = image_series.get_image_filename(
                    k_frame = k_frame                ,
                    suffix  = direction+stack_suffix ) ,
                verbose  = verbose-1                   ) ,
            verbose = verbose-1) for direction in directions]

        image_combined = myvtk.createImageFromSizeAndRes(
            dim  = images["n_dim"]    ,
            size = images["L"]        ,
            res  = images["n_voxels"] )
        scalars_combined = image_combined.GetPointData().GetScalars()

        for k_point in range(image_combined.GetNumberOfPoints()):
            image_combined.GetPoint(k_point, x)
            I = 1.
            for interpolator in interpolators:
                interpolator.Interpolate(x, i)
                I *= max(i[0], 0.) # the stacks may undershoot, which must not be folded into the signal
            scalars_combined.SetTuple1(k_point, I**(1./images["n_dim"]))

        myvtk.writeImage(
            image    = image_combined                                   ,
            filename = image_series.get_image_filename(k_frame=k_frame) ,
            verbose  = verbose-1                                        )

    if not (keep_temporary_images):
        remove_stacks(
            image_series       = image_series       ,
            directions         = directions         ,
            stack_suffix       = stack_suffix       ,
            n_frames_generated = images_n_frames    ,
            n_frames           = images["n_frames"] )

################################################################################

def generate_undersampled_images_through_Fourier_interpolation(
        images                           ,
        structure                        ,
        texture                          ,
        noise                            ,
        deformation                      ,
        evolution                        ,
        undersampling_level              ,
        stacks_normalization  = "linear" , # "linear" or "clip"
        keep_temporary_images = 0        ,
        verbose               = 0        ):

    mypy.my_print(verbose, "*** generate_undersampled_images_through_Fourier_interpolation ***")

    assert (stacks_normalization in ("clip", "linear")),\
        "\"stacks_normalization\" (=\""+str(stacks_normalization)+"\") must be \"clip\" or \"linear\". Aborting."

    images_basename             = images["basename"]
    images_n_frames             = images["n_frames"]
    images_downsampling_factors = images["downsampling_factors"][:] if ("downsampling_factors" in images) else None
    texture_type                = texture["type"]

    temporal_downsampling_factor = images["temporal_downsampling_factor"] if ("temporal_downsampling_factor" in images) else 1
    temporal_window_size         = images["temporal_window_size"]         if ("temporal_window_size"         in images) else None
    images["temporal_downsampling_factor"] = 1 # the temporal downsampling is applied to each stack below, once it has been generated
    images["temporal_window_size"]         = None

    directions = ["X", "Y", "Z"][:images["n_dim"]]

    # Each stack is generated at full resolution, and then actually undersampled onto a coarser grid in the directions orthogonal to its tagging direction—this is what is acquired—before being interpolated back onto the full grid in Fourier space, i.e., by zero padding its k-space.
    # Note that the downsampling and upsampling both leave the image origin untouched, so that all the stacks remain on the very same grid and can simply be combined point by point afterwards.
    # (This would not be the case if each stack were generated directly onto its own coarse grid, as the grid origin is then half a coarse voxel instead of half a fine one.)
    for k_dim, direction in enumerate(directions):
        images["basename"]             = images_basename+"-"+direction
        images["n_frames"]             = images_n_frames
        images["downsampling_factors"] = [1]*images["n_dim"] # the undersampling is performed explicitly below
        texture["type"]                = "tagg"+direction

        dwarp.generate_images(
            images      = images      ,
            structure   = structure   ,
            texture     = texture     ,
            noise       = noise       ,
            deformation = deformation ,
            evolution   = evolution   ,
            verbose     = verbose-1   )

        dwarp.compute_downsampled_images(
            images_folder        = images["folder"]                                              ,
            images_basename      = images["basename"]                                            ,
            downsampling_factors = get_undersampling_factors(images, k_dim, undersampling_level) ,
            images_ext           = images["ext"]                                                 ,
            keep_resolution      = 0                                                             ,
            verbose              = verbose-1                                                     )

    images["basename"]                     = images_basename
    images["n_frames"]                     = images_n_frames
    images["temporal_downsampling_factor"] = temporal_downsampling_factor
    images["temporal_window_size"]         = temporal_window_size
    if (images_downsampling_factors is None):
        del images["downsampling_factors"]
    else:
        images["downsampling_factors"] = images_downsampling_factors
    texture["type"]    = texture_type

    stack_suffix = temporally_downsample_stacks(
        images                       = images                       ,
        directions                   = directions                   ,
        temporal_downsampling_factor = temporal_downsampling_factor ,
        temporal_window_size         = temporal_window_size         ,
        verbose                      = verbose                      )

    for k_dim, direction in enumerate(directions): # Fourier interpolation back onto the full grid
        dwarp.compute_upsampled_images(
            images_folder      = images["folder"]                                              ,
            images_basename    = images["basename"]+"-"+direction+stack_suffix                 ,
            upsampling_factors = get_undersampling_factors(images, k_dim, undersampling_level) ,
            images_ext         = images["ext"]                                                 ,
            verbose            = verbose-1                                                     )

    if (stacks_normalization == "linear"):
        normalize_stacks(
            images       = images       ,
            directions   = directions   ,
            stack_suffix = stack_suffix ,
            verbose      = verbose      )

    image_series = dwarp.ImageSeries( # the combined images do not exist yet, so the series is declared, not read
        folder   = images["folder"]   ,
        basename = images["basename"] ,
        n_frames = images["n_frames"] ,
        zfill    = images["zfill"]    ,
        ext      = images["ext"]      ,
        verbose  = 0                  )

    if   (images["ext"] == "vtk"):
        reader_type = vtk.vtkImageReader
    elif (images["ext"] == "vti"):
        reader_type = vtk.vtkXMLImageDataReader
    else:
        assert 0, "\"ext\" must be \".vtk\" or \".vti\". Aborting."
    reader = reader_type()

    reader.SetFileName(image_series.get_image_filename(k_frame=0, suffix=directions[0]+stack_suffix))
    reader.Update()
    image_combined = vtk.vtkImageData()
    image_combined.DeepCopy(reader.GetOutput())
    scalars_combined_vtk = image_combined.GetPointData().GetScalars()
    scalars_combined_np = vtk.util.numpy_support.vtk_to_numpy(scalars_combined_vtk)

    prod_scalars  = numpy.empty(scalars_combined_np.shape) # the stacks may be of integer type, so they cannot be combined in place
    stack_scalars = numpy.empty(scalars_combined_np.shape)

    for k_frame in range(images["n_frames"]):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        prod_scalars[:] = 1.
        for direction in directions:
            reader.SetFileName(image_series.get_image_filename(k_frame=k_frame, suffix=direction+stack_suffix))
            reader.Update()
            numpy.clip(vtk.util.numpy_support.vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars()), 0., None, out=stack_scalars) # the Fourier interpolation may undershoot, which must not be folded into the signal
            prod_scalars *= stack_scalars

        prod_scalars **= 1./images["n_dim"]
        scalars_combined_np[:] = prod_scalars
        scalars_combined_vtk.Modified()
        image_combined.Modified()

        myvtk.writeImage(
            image    = image_combined                                  ,
            filename = image_series.get_image_filename(k_frame=k_frame),
            verbose  = verbose-1                                       )

    if not (keep_temporary_images):
        remove_stacks(
            image_series       = image_series      ,
            directions         = directions        ,
            stack_suffix       = stack_suffix      ,
            n_frames_generated = images_n_frames   ,
            n_frames           = images["n_frames"])

################################################################################

generate_undersampled_images = generate_undersampled_images_through_Fourier_interpolation

if (__name__ == "__main__"):
    import fire
    fire.Fire(generate_undersampled_images)
