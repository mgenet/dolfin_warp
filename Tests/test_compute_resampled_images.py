#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2026                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

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

tol = 1e-12

def get_image(basename, k_frame, suffix=None):
    image_series = dwarp.ImageSeries(
        folder   = res_folder,
        basename = basename  ,
        verbose  = 0         )
    return myvtk.readImage(
        filename=image_series.get_image_filename(k_frame=k_frame, suffix=suffix),
        verbose=0)

def get_scalars(basename, k_frame, suffix=None):
    return vtk.util.numpy_support.vtk_to_numpy(
        get_image(basename, k_frame, suffix).GetPointData().GetScalars()).astype(float)

################################################################################
###                                                       Spatial resampling ###
################################################################################

n_dim_lst  = [ ]
n_dim_lst += [2]
# n_dim_lst += [3]

n_voxels_lst  = [  ]
n_voxels_lst += [11] # odd
n_voxels_lst += [12] # even

resampling_factor_lst  = [ ]
resampling_factor_lst += [2]
resampling_factor_lst += [3]

n_frames = 3

for n_dim              in n_dim_lst             :
 for n_voxels           in n_voxels_lst          :
  for resampling_factor  in resampling_factor_lst :

    print("n_dim = "+str(n_dim))
    print("n_voxels = "+str(n_voxels))
    print("resampling_factor = "+str(resampling_factor))

    images_basename  = str(n_dim)+"D"
    images_basename += "-"+str(n_voxels)
    images_basename += "-"+str(resampling_factor)

    images = {
        "n_dim"     : n_dim            ,
        "L"         : [1.]*n_dim       ,
        "n_voxels"  : [n_voxels]*n_dim ,
        "T"         : 1.               ,
        "n_frames"  : n_frames         ,
        "data_type" : "float"          ,
        "folder"    : res_folder       ,
        "basename"  : images_basename  }

    dwarp.generate_images(
        images      = images                                                   ,
        structure   = {"type":"box", "Xmin":[0.25]*n_dim, "Xmax":[0.75]*n_dim} ,
        texture     = {"type":"tagging", "s":0.25}                             ,
        noise       = {"type":"no"}                                            ,
        deformation = {"type":"no"}                                            ,
        evolution   = {"type":"linear"}                                        ,
        verbose     = 0                                                        )

    dwarp.compute_upsampled_images(
        images_folder      = res_folder                ,
        images_basename    = images_basename           ,
        upsampling_factors = [resampling_factor]*n_dim ,
        suffix             = "upsampled"               ,
        verbose            = 0                         )

    dwarp.compute_downsampled_images(
        images_folder        = res_folder                   ,
        images_basename      = images_basename+"-upsampled" ,
        downsampling_factors = [resampling_factor]*n_dim    ,
        keep_resolution      = 0                            ,
        suffix               = "downsampled"                ,
        verbose              = 0                            )

    dwarp.compute_downsampled_images(
        images_folder        = res_folder                ,
        images_basename      = images_basename           ,
        downsampling_factors = [resampling_factor]*n_dim ,
        keep_resolution      = 1                         ,
        suffix               = "cropped"                 ,
        verbose              = 0                         )

    dwarp.compute_downsampled_images(
        images_folder        = res_folder                 ,
        images_basename      = images_basename+"-cropped" ,
        downsampling_factors = [resampling_factor]*n_dim  ,
        keep_resolution      = 1                          ,
        suffix               = "twice"                    ,
        verbose              = 0                          )

    for k_frame in range(n_frames):

        image            = get_image(images_basename, k_frame             )
        image_upsampled  = get_image(images_basename, k_frame, "upsampled")
        image_cropped    = get_image(images_basename, k_frame, "cropped"  )

        # upsampling multiplies the number of voxels, and divides the voxel size, by the upsampling factor
        assert (list(image_upsampled.GetDimensions())[:n_dim] == [n_voxels*resampling_factor]*n_dim),\
            "Wrong upsampled dimensions ("+str(image_upsampled.GetDimensions())+"). Aborting."
        assert (numpy.allclose(numpy.multiply(image_upsampled.GetSpacing()[:n_dim], resampling_factor), image.GetSpacing()[:n_dim])),\
            "Wrong upsampled spacing ("+str(image_upsampled.GetSpacing())+"). Aborting."

        # cropping the k-space at constant resolution leaves the grid untouched
        assert (image_cropped.GetDimensions() == image.GetDimensions()),\
            "Cropping should not change the dimensions ("+str(image_cropped.GetDimensions())+"). Aborting."
        assert (numpy.allclose(image_cropped.GetOrigin(), image.GetOrigin())),\
            "Cropping should not change the origin ("+str(image_cropped.GetOrigin())+"). Aborting."

        scalars             = get_scalars(images_basename, k_frame                         )
        scalars_upsampled   = get_scalars(images_basename, k_frame, "upsampled"            )
        scalars_downsampled = get_scalars(images_basename, k_frame, "upsampled-downsampled")
        scalars_cropped     = get_scalars(images_basename, k_frame, "cropped"              )
        scalars_twice       = get_scalars(images_basename, k_frame, "cropped-twice"        )

        # upsampling and downsampling back is the identity, as zero padding and cropping the k-space are
        error = numpy.max(numpy.abs(scalars_downsampled-scalars))
        print("    k_frame = "+str(k_frame)+", upsampling/downsampling error = "+str(error))
        assert (error < tol),\
            "Upsampling then downsampling should be the identity (error = "+str(error)+"). Aborting."

        # none of these operations touches the zero frequency, so the mean is preserved
        for name, resampled_scalars in [("upsampled"  , scalars_upsampled  ),
                                        ("downsampled", scalars_downsampled),
                                        ("cropped"    , scalars_cropped    )]:
            error = abs(numpy.mean(resampled_scalars)-numpy.mean(scalars))
            assert (error < tol),\
                "The mean should be preserved by the "+name+" images (error = "+str(error)+"). Aborting."

        # cropping the same frequencies twice is the same as cropping them once
        error = numpy.max(numpy.abs(scalars_twice-scalars_cropped))
        assert (error < tol),\
            "Cropping the k-space should be idempotent (error = "+str(error)+"). Aborting."

################################################################################
###                                                      Temporal resampling ###
################################################################################

# Each frame is considered to cover [k-1/2, k+1/2], and that support is redistributed over ceil(n_frames/temporal_downsampling_factor) frames of equal width, so that the downsampled frames are weighted averages of the frames, with the weights given by the overlap of their supports.
temporal_test_lst  = [ ]
temporal_test_lst += [(5, 2, None , [[3/5, 2/5,   0,   0,   0  ],
                                     [  0, 1/5, 3/5, 1/5,   0  ],
                                     [  0,   0,   0, 2/5, 3/5  ]])]
temporal_test_lst += [(5, 3, None , [[2/5, 2/5, 1/5,   0,   0  ],
                                     [  0,   0, 1/5, 2/5, 2/5  ]])]
temporal_test_lst += [(6, 2, None , [[1/2, 1/2,   0,   0,   0, 0],
                                     [  0,   0, 1/2, 1/2,   0, 0],
                                     [  0,   0,   0,   0, 1/2, 1/2]])]
temporal_test_lst += [(5, 1, None , numpy.eye(5).tolist())]
temporal_test_lst += [(5, 2, 10./3, [[2/5 , 2/5 , 1/5 ,   0  ,   0  ],
                                     [1/20, 3/10, 3/10, 3/10, 1/20 ],
                                     [  0 ,   0 , 1/5 , 2/5 , 2/5  ]])]

for n_frames, temporal_downsampling_factor, temporal_window_size, weights_ref in temporal_test_lst:

    print("n_frames = "+str(n_frames))
    print("temporal_downsampling_factor = "+str(temporal_downsampling_factor))
    print("temporal_window_size = "+str(temporal_window_size))

    # A basename of its own for each case, so that the frames of a previous, longer case are not picked up
    images_basename  = "temporal-"+str(n_frames)
    images_basename += "-"+str(temporal_downsampling_factor)
    images_basename += "-"+str(temporal_window_size)

    # The weights are obtained one at a time, by downsampling a series where a single frame is one and all the others are zero.
    weights = [[0.]*n_frames for k_frame_downsampled in range(len(weights_ref))]
    for k_frame_one in range(n_frames):

        for k_frame in range(n_frames):
            image = myvtk.createImageFromSizeAndRes(
                dim  = 2  ,
                size = 1. ,
                res  = 2  )
            image.GetPointData().GetScalars().FillComponent(0, 1. if (k_frame == k_frame_one) else 0.)
            myvtk.writeImage(
                image    = image                                                           ,
                filename = res_folder+"/"+images_basename+"_"+str(k_frame).zfill(2)+".vti" ,
                verbose  = 0                                                               )

        dwarp.compute_temporally_downsampled_images(
            images_folder                = res_folder                   ,
            images_basename              = images_basename              ,
            temporal_downsampling_factor = temporal_downsampling_factor ,
            temporal_window_size         = temporal_window_size         ,
            suffix                       = "downsampled"                ,
            verbose                      = 0                            )

        for k_frame_downsampled in range(len(weights_ref)):
            weights[k_frame_downsampled][k_frame_one] = get_scalars(images_basename, k_frame_downsampled, "downsampled")[0]

    for k_frame_downsampled in range(len(weights_ref)):
        print("    frame "+str(k_frame_downsampled)+" = "+"+".join(
            str(round(weight, 6))+"*im"+str(k_frame) for k_frame, weight in enumerate(weights[k_frame_downsampled]) if (abs(weight) > tol)))

        # the weights are the overlaps of the frame supports with the downsampled frame support
        error = numpy.max(numpy.abs(numpy.subtract(weights[k_frame_downsampled], weights_ref[k_frame_downsampled])))
        assert (error < tol),\
            "Wrong weights for frame "+str(k_frame_downsampled)+" ("+str(weights[k_frame_downsampled])+" instead of "+str(weights_ref[k_frame_downsampled])+"). Aborting."

        # the weights are normalized, so that a constant series stays constant
        error = abs(numpy.sum(weights[k_frame_downsampled])-1.)
        assert (error < tol),\
            "The weights of frame "+str(k_frame_downsampled)+" should sum to one (error = "+str(error)+"). Aborting."

shutil.rmtree(res_folder, ignore_errors=1)
