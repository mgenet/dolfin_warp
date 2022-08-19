#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import numpy
import os
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def generateUndersampledImages(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        undersampling_level,
        keep_temporary_images=0,
        verbose=0):

    mypy.my_print(verbose, "*** generateUndersampledImages ***")

    images_basename = images["basename"]
    images_n_voxels = images["n_voxels"][:]

    images["n_voxels"][1] /= undersampling_level
    if (images["n_dim"] == 3):
        images["n_voxels"][2] /= undersampling_level
    images["basename"] = images_basename+"-X"
    texture["type"] = "taggX"
    dwarp.generate_images(
        images=images,
        structure=structure,
        texture=texture,
        noise=noise,
        deformation=deformation,
        evolution=evolution,
        verbose=verbose-1)
    images["n_voxels"][:] = images_n_voxels[:]
    images["basename"] = images_basename

    images["n_voxels"][0] /= undersampling_level
    if (images["n_dim"] == 3):
        images["n_voxels"][2] /= undersampling_level
    images["basename"] = images_basename+"-Y"
    texture["type"] = "taggY"
    dwarp.generate_images(
        images=images,
        structure=structure,
        texture=texture,
        noise=noise,
        deformation=deformation,
        evolution=evolution,
        verbose=verbose-1)
    images["n_voxels"][:] = images_n_voxels[:]
    images["basename"] = images_basename

    if (images["n_dim"] == 3):
        images["n_voxels"][0] /= undersampling_level
        images["n_voxels"][1] /= undersampling_level
        images["basename"] = images_basename+"-Z"
        texture["type"] = "taggZ"
        dwarp.generate_images(
            images=images,
            structure=structure,
            texture=texture,
            noise=noise,
            deformation=deformation,
            evolution=evolution,
            verbose=verbose-1)
        images["n_voxels"][:] = images_n_voxels[:]
        images["basename"] = images_basename

    if ("zfill" not in images.keys()):
        images["zfill"] = len(str(images["n_frames"]))
    for k_frame in range(images["n_frames"]):
        imageX = myvtk.readImage(
            filename=images["folder"]+"/"+images["basename"]+"-X_"+str(k_frame).zfill(images["zfill"])+".vti",
            verbose=verbose-1)
        interpolatorX = myvtk.getImageInterpolator(
            image=imageX,
            verbose=verbose-1)
        iX = numpy.empty(1)

        imageY = myvtk.readImage(
            filename=images["folder"]+"/"+images["basename"]+"-Y_"+str(k_frame).zfill(images["zfill"])+".vti",
            verbose=verbose-1)
        interpolatorY = myvtk.getImageInterpolator(
            image=imageY,
            verbose=verbose-1)
        iY = numpy.empty(1)

        if (images["n_dim"] == 2):
            imageXY = myvtk.createImage(
                origin=[images["L"][0]/images["n_voxels"][0]/2, images["L"][1]/images["n_voxels"][1]/2, 0.],
                spacing=[images["L"][0]/images["n_voxels"][0], images["L"][1]/images["n_voxels"][1], 1.],
                extent=[0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0, 0])
            scalars = imageXY.GetPointData().GetScalars()
            x = numpy.empty(3)
            for k_point in range(imageXY.GetNumberOfPoints()):
                imageXY.GetPoint(k_point, x)
                interpolatorX.Interpolate(x, iX)
                interpolatorY.Interpolate(x, iY)
                scalars.SetTuple(k_point, (iX*iY)**(1./2))
            myvtk.writeImage(
                image=imageXY,
                filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+".vti",
                verbose=verbose-1)
            if not (keep_temporary_images):
                os.system("rm "+images["folder"]+"/"+images["basename"]+"-X_"+str(k_frame).zfill(images["zfill"])+".vti")
                os.system("rm "+images["folder"]+"/"+images["basename"]+"-Y_"+str(k_frame).zfill(images["zfill"])+".vti")

        elif (images["n_dim"] == 3):
            imageZ = myvtk.readImage(
                filename=images["folder"]+"/"+images["basename"]+"-Z_"+str(k_frame).zfill(images["zfill"])+".vti",
                verbose=verbose-1)
            interpolatorZ = myvtk.getImageInterpolator(
                image=imageZ,
                verbose=verbose-1)
            iZ = numpy.empty(1)

            imageXYZ = myvtk.createImage(
                origin=[images["L"][0]/images["n_voxels"][0]/2, images["L"][1]/images["n_voxels"][1]/2, images["L"][2]/images["n_voxels"][2]/2],
                spacing=[images["L"][0]/images["n_voxels"][0], images["L"][1]/images["n_voxels"][1], images["L"][2]/images["n_voxels"][2]],
                extent=[0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0, images["n_voxels"][2]-1])
            scalars = imageXYZ.GetPointData().GetScalars()
            x = numpy.empty(3)
            for k_point in range(imageXYZ.GetNumberOfPoints()):
                imageXYZ.GetPoint(k_point, x)
                interpolatorX.Interpolate(x, iX)
                interpolatorY.Interpolate(x, iY)
                interpolatorZ.Interpolate(x, iZ)
                scalars.SetTuple(k_point, (iX*iY*iZ)**(1./3))
            myvtk.writeImage(
                image=imageXYZ,
                filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+".vti",
                verbose=verbose-1)
            if not (keep_temporary_images):
                os.system("rm "+images["folder"]+"/"+images["basename"]+"-X_"+str(k_frame).zfill(images["zfill"])+".vti")
                os.system("rm "+images["folder"]+"/"+images["basename"]+"-Y_"+str(k_frame).zfill(images["zfill"])+".vti")
                os.system("rm "+images["folder"]+"/"+images["basename"]+"-Z_"+str(k_frame).zfill(images["zfill"])+".vti")
