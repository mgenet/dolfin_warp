#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import math
import numpy
import os
import shutil
import sys
import vtk.util.numpy_support

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk
import dolfin_warp        as dwarp

################################################################################

res_folder = sys.argv[0][:-3]
if not os.path.exists(res_folder): os.mkdir(res_folder)

# test = mypy.Test(
#     res_folder=res_folder,
#     perform_tests=0,
#     clean_after_tests=1)

tol = 1e-9
# the linear interpolation goes through vtkImageInterpolator, and the normalization through
# vtkImageShiftScale with a float output, both of which only work in single precision
tol_single_precision = 1e-6

def get_image(basename, k_frame, suffix=None):
    image_series = dwarp.ImageSeries(
        folder   = res_folder ,
        basename = basename   ,
        verbose  = 0          )
    return myvtk.readImage(
        filename = image_series.get_image_filename(
            k_frame = k_frame ,
            suffix  = suffix  ) ,
        verbose  = 0            )

def get_scalars(basename, k_frame, suffix=None):
    return vtk.util.numpy_support.vtk_to_numpy(
        get_image(basename, k_frame, suffix).GetPointData().GetScalars()).astype(float)

def get_params(n_dim, n_voxels, n_frames, basename, **kwargs):
    images = {
        "n_dim"     : n_dim            ,
        "L"         : [1.]*n_dim       ,
        "n_voxels"  : [n_voxels]*n_dim ,
        "T"         : 1.               ,
        "n_frames"  : n_frames         ,
        "data_type" : "float"          ,
        "folder"    : res_folder       ,
        "basename"  : basename         }
    images.update(kwargs)
    return (images                                                 ,
            {"type":"box", "Xmin":[0.2]*n_dim, "Xmax":[0.8]*n_dim} , # not aligned with the grids
            {"type":"tagging", "s":0.25}                           ,
            {"type":"no"}                                          ,
            {"type":"translation", "Dx":0.1, "Dy":0.}              ,
            {"type":"linear"}                                      )

# the Fourier implementation interpolates the stacks back onto the full grid, so that they can be combined point by point, while the linear implementation keeps them on their own coarse grids and interpolates them while combining
implementation_lst  = [ ]
implementation_lst += [("Fourier", dwarp.generate_undersampled_images_through_Fourier_interpolation, tol              , True )]
implementation_lst += [("linear" , dwarp.generate_undersampled_images_through_linear_interpolation , tol_single_precision, False)]

################################################################################
### Without undersampling, the stacks recombine into the tagged image        ###
################################################################################

# The tagging texture combines the directions multiplicatively, i.e., exactly as the stacks are combined here, so with an undersampling level of one the result must be the tagged image itself.
# Note that this requires the stacks not to be normalized, as normalizing them rescales each of them by its own extrema, which does not preserve the tagged image (unless the stacks already reach one).

n_dim_lst  = [ ]
n_dim_lst += [2]
# n_dim_lst += [3]

for n_dim in n_dim_lst:

    n_voxels = 12 if (n_dim == 2) else 8
    n_frames = 2

    images, structure, texture, noise, deformation, evolution = get_params(n_dim, n_voxels, n_frames, "reference")
    dwarp.generate_images(
        images      = images      ,
        structure   = structure   ,
        texture     = texture     ,
        noise       = noise       ,
        deformation = deformation ,
        evolution   = evolution   ,
        verbose     = 0           )

    for implementation_name, generate_undersampled_images, implementation_tol, stacks_on_full_grid in implementation_lst:

        print("n_dim = "+str(n_dim))
        print("implementation = "+implementation_name)
        print("undersampling_level = 1")

        images, structure, texture, noise, deformation, evolution = get_params(n_dim, n_voxels, n_frames, "level1-"+implementation_name)
        generate_undersampled_images(
            images               = images      ,
            structure            = structure   ,
            texture              = texture     ,
            noise                = noise       ,
            deformation          = deformation ,
            evolution            = evolution   ,
            undersampling_level  = 1           ,
            stacks_normalization = "clip"      ,
            verbose              = 0           )

        for k_frame in range(n_frames):
            error = numpy.max(numpy.abs(get_scalars("level1-"+implementation_name, k_frame)-get_scalars("reference", k_frame)))
            print("    k_frame = "+str(k_frame)+", error = "+str(error))
            assert (error < implementation_tol),\
                "Without undersampling the combined images should be the tagged images (error = "+str(error)+"). Aborting."

        # the combined images are on the very same grid as the tagged images
        image           = get_image("reference"                    , 0)
        image_undersampled = get_image("level1-"+implementation_name, 0)
        assert (image_undersampled.GetDimensions() == image.GetDimensions()),\
            "Wrong dimensions ("+str(image_undersampled.GetDimensions())+"). Aborting."
        assert (numpy.allclose(image_undersampled.GetOrigin(), image.GetOrigin())),\
            "Wrong origin ("+str(image_undersampled.GetOrigin())+"). Aborting."
        assert (numpy.allclose(image_undersampled.GetSpacing(), image.GetSpacing())),\
            "Wrong spacing ("+str(image_undersampled.GetSpacing())+"). Aborting."

################################################################################
### With undersampling                                                       ###
################################################################################

n_dim               = 2
n_voxels            = 12
n_frames            = 6
undersampling_level = 2

stacks_normalization_lst  = [        ]
stacks_normalization_lst += ["linear"]
stacks_normalization_lst += ["clip"  ]

temporal_downsampling_factor_lst  = [ ]
temporal_downsampling_factor_lst += [1]
temporal_downsampling_factor_lst += [3]

for implementation_name, generate_undersampled_images, implementation_tol, stacks_on_full_grid in implementation_lst               :
 for stacks_normalization                                                                       in stacks_normalization_lst         :
  for temporal_downsampling_factor                                                               in temporal_downsampling_factor_lst :

    print("implementation = "+implementation_name)
    print("stacks_normalization = "+stacks_normalization)
    print("temporal_downsampling_factor = "+str(temporal_downsampling_factor))

    images_basename  = implementation_name
    images_basename += "-"+stacks_normalization
    images_basename += "-"+str(temporal_downsampling_factor)

    images, structure, texture, noise, deformation, evolution = get_params(n_dim, n_voxels, n_frames, images_basename, temporal_downsampling_factor=temporal_downsampling_factor)

    generate_undersampled_images(
        images                = images               ,
        structure             = structure            ,
        texture               = texture              ,
        noise                 = noise                ,
        deformation           = deformation          ,
        evolution             = evolution            ,
        undersampling_level   = undersampling_level  ,
        stacks_normalization  = stacks_normalization ,
        keep_temporary_images = 1                    ,
        verbose               = 0                    )

    # the number of frames follows the temporal downsampling, and is reported back to the caller
    n_frames_downsampled = math.ceil(n_frames/temporal_downsampling_factor)
    assert (images["n_frames"] == n_frames_downsampled),\
        "Wrong number of frames ("+str(images["n_frames"])+" instead of "+str(n_frames_downsampled)+"). Aborting."
    assert (len(glob.glob(res_folder+"/"+images_basename+"_[0-9]*.vti")) == n_frames_downsampled),\
        "Wrong number of images. Aborting."

    # the parameters of the caller are restored
    assert (images["basename"] == images_basename),\
        "\"basename\" should be restored ("+str(images["basename"])+"). Aborting."
    assert (images["n_voxels"] == [n_voxels]*n_dim),\
        "\"n_voxels\" should be restored ("+str(images["n_voxels"])+"). Aborting."
    assert (texture["type"] == "tagging"),\
        "The texture type should be restored ("+str(texture["type"])+"). Aborting."

    stack_suffix = "-tdown="+str(temporal_downsampling_factor) if (temporal_downsampling_factor != 1) else ""

    for k_frame in range(n_frames_downsampled):

        scalars = get_scalars(images_basename, k_frame)

        # the images are combined from the stacks, which are all nonnegative
        assert (numpy.min(scalars) > -tol),\
            "The combined images should be nonnegative (min = "+str(numpy.min(scalars))+"). Aborting."

        if (stacks_on_full_grid):
            # the combined images are the geometric mean of the stacks, which are the ones that have been
            # temporally downsampled, i.e., the stacks are combined after, and not before, being averaged
            product = numpy.ones(numpy.shape(scalars))
            for direction in ["X", "Y", "Z"][:n_dim]:
                product *= numpy.clip(get_scalars(images_basename, k_frame, direction+stack_suffix), 0., None)
            error = numpy.max(numpy.abs(product**(1./n_dim)-scalars))
            assert (error < (tol if (stacks_normalization == "clip") else tol_single_precision)),\
                "The combined images should be the geometric mean of the stacks (error = "+str(error)+"). Aborting."
        else:
            # the stacks are kept on their own grids, coarse in the directions orthogonal to their own
            for k_dim, direction in enumerate(["X", "Y", "Z"][:n_dim]):
                dimensions_ref = [n_voxels if (k == k_dim) else n_voxels//undersampling_level for k in range(n_dim)]
                dimensions     = list(get_image(images_basename, k_frame, direction+stack_suffix).GetDimensions())[:n_dim]
                assert (dimensions == dimensions_ref),\
                    "Wrong dimensions for stack "+direction+" ("+str(dimensions)+" instead of "+str(dimensions_ref)+"). Aborting."

    # the undersampling must have an effect
    images, structure, texture, noise, deformation, evolution = get_params(n_dim, n_voxels, n_frames_downsampled, images_basename+"-notundersampled")
    dwarp.generate_images(
        images      = images      ,
        structure   = structure   ,
        texture     = texture     ,
        noise       = noise       ,
        deformation = deformation ,
        evolution   = evolution   ,
        verbose     = 0           )
    difference = numpy.max(numpy.abs(get_scalars(images_basename, 0)-get_scalars(images_basename+"-notundersampled", 0)))
    print("    difference with the not undersampled images = "+str(difference))
    assert (difference > 1e-2),\
        "The undersampling should have an effect (difference = "+str(difference)+"). Aborting."

################################################################################
### The temporary images are removed unless they are asked for               ###
################################################################################

images, structure, texture, noise, deformation, evolution = get_params(2, 12, 4, "temporary", temporal_downsampling_factor=2)
dwarp.generate_undersampled_images(
    images                = images      ,
    structure             = structure   ,
    texture               = texture     ,
    noise                 = noise       ,
    deformation           = deformation ,
    evolution             = evolution   ,
    undersampling_level   = 2           ,
    keep_temporary_images = 0           ,
    verbose               = 0           )
leftovers = glob.glob(res_folder+"/temporary-*")
print("leftover temporary images: "+str(leftovers))
assert (len(leftovers) == 0),\
    "The temporary images should have been removed ("+str(leftovers)+"). Aborting."

shutil.rmtree(res_folder, ignore_errors=1)
