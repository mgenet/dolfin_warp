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
import shutil
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp
from .generate_images_Image   import Image
from .generate_images_Mapping import Mapping

################################################################################

def set_I_woGrad(
        image,
        X,
        I,
        scalars,
        k_point,
        G=None,
        Finv=None,
        vectors=None):

    image.I0(X, I)
    scalars.SetTuple(k_point, I)

def set_I_wGrad(
        image,
        X,
        I,
        scalars,
        k_point,
        G,
        Finv,
        vectors):

    image.I0_wGrad(X, I, G)
    scalars.SetTuple(k_point, I)
    G = numpy.dot(G, Finv)
    vectors.SetTuple(k_point, G)

################################################################################

def generate_images(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        generate_gradient=0,
        keep_temp_images=0,
        verbose=0):

    mypy.my_print(verbose, "*** generate_images ***")

    assert ("n_integration" not in images),\
        "\"n_integration\" has been deprecated. Use \"upsampling\" instead. Aborting."

    if ("upsampling_factors" not in images):
        images["upsampling_factors"] = [1]*images["n_dim"]
    if ("downsampling_factors" not in images):
        images["downsampling_factors"] = [1]*images["n_dim"]
    if ("zfill" not in images):
        images["zfill"] = len(str(images["n_frames"]))
    if ("ext" not in images):
        images["ext"] = "vti"

    if not os.path.exists(images["folder"]):
        os.mkdir(images["folder"])

    for filename in glob.glob(images["folder"]+"/"+images["basename"]+"_*.*"):
        os.remove(filename)

    image = Image(
        images,
        structure,
        texture,
        noise,
        generate_gradient)
    mapping = Mapping(
        images,
        structure,
        deformation,
        evolution,
        generate_gradient)

    vtk_image = vtk.vtkImageData()

    n_voxels           =                     images["n_voxels"]
    n_voxels_upsampled = list(numpy.multiply(images["n_voxels"], images["upsampling_factors"]))

    dimensions           = n_voxels          +[1]*(3-images["n_dim"])
    dimensions_upsampled = n_voxels_upsampled+[1]*(3-images["n_dim"])
    mypy.my_print(verbose, "dimensions_upsampled = "+str(dimensions_upsampled))
    vtk_image.SetDimensions(dimensions_upsampled)

    delta           = list(numpy.divide(images["L"], n_voxels          ))
    delta_upsampled = list(numpy.divide(images["L"], n_voxels_upsampled))

    spacing           = delta          +[1.]*(3-images["n_dim"])
    spacing_upsampled = delta_upsampled+[1.]*(3-images["n_dim"])
    mypy.my_print(verbose, "spacing_upsampled = "+str(spacing_upsampled))
    vtk_image.SetSpacing(spacing_upsampled)

    origin           = list(numpy.divide(delta, 2))+[0.]*(3-images["n_dim"])
    origin_upsampled = origin
    mypy.my_print(verbose, "origin_upsampled = "+str(origin_upsampled))
    vtk_image.SetOrigin(origin_upsampled)

    n_points_upsampled = vtk_image.GetNumberOfPoints()
    vtk_scalars = myvtk.createDoubleArray(
        name="ImageScalars",
        n_components=1,
        n_tuples=n_points_upsampled,
        verbose=verbose-1)
    vtk_image.GetPointData().SetScalars(vtk_scalars)

    if (generate_gradient):
        vtk_gradient = vtk.vtkImageData()
        vtk_gradient.DeepCopy(vtk_image)

        vtk_vectors = myvtk.createFloatArray(
            name="ImageScalarsGradient",
            n_components=3,
            n_tuples=n_points_upsampled,
            verbose=verbose-1)
        vtk_gradient.GetPointData().SetScalars(vtk_vectors)
    else:
        vtk_gradient = None
        vtk_vectors  = None

    x = numpy.empty(3)
    X = numpy.empty(3)
    I = numpy.empty(1)
    global_min = float("+Inf")
    global_max = float("-Inf")
    if (generate_gradient):
        G     = numpy.empty(3)
        Finv  = numpy.empty((3,3))
        set_I = set_I_wGrad
    else:
        G     = None
        Finv  = None
        set_I = set_I_woGrad

    for k_frame in range(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mypy.my_print(verbose, "t = "+str(t))
        mapping.init_t(t)
        for k_point in range(n_points_upsampled):
            vtk_image.GetPoint(k_point, x)
            #print("x0 = "+str(x))
            mapping.X(x, X, Finv)
            #print("X = "+str(X))
            set_I(image, X, I, vtk_scalars, k_point, G, Finv, vtk_vectors)
            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
        #print(vtk_image)
        myvtk.writeImage(
            image=vtk_image,
            filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
            verbose=verbose-1)
        if (generate_gradient):
            #print(vtk_gradient)
            myvtk.writeImage(
                image=vtk_gradient,
                filename=images["folder"]+"/"+images["basename"]+"-grad"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                verbose=verbose-1)
    # mypy.my_print(verbose, "global_min = "+str(global_min))
    # mypy.my_print(verbose, "global_max = "+str(global_max))

    if (images["upsampling_factors"] == [1]*images["n_dim"]):
        downsampling = False
    else:
        downsampling = True

        if (keep_temp_images):
            for k_frame in range(images["n_frames"]):
                shutil.copy(
                    src=images["folder"]+"/"+images["basename"]             +"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                    dst=images["folder"]+"/"+images["basename"]+"_upsampled"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])

        dwarp.compute_downsampled_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            downsampling_factors=images["upsampling_factors"],
            keep_resolution=0,
            verbose=verbose)

    if (images["downsampling_factors"] == [1]*images["n_dim"]):
        downsampling = False
    else:
        downsampling = True

        if (keep_temp_images):
            for k_frame in range(images["n_frames"]):
                shutil.copy(
                    src=images["folder"]+"/"+images["basename"]                  +"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                    dst=images["folder"]+"/"+images["basename"]+"_predownsampled"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])

        dwarp.compute_downsampled_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            downsampling_factors=images["downsampling_factors"],
            keep_resolution=1,
            verbose=verbose)

    if (images["data_type"] in ("float")):
        normalizing = False
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float", "uint8", "uint16", "uint32", "uint64", "ufloat")):
        normalizing = True

        if (keep_temp_images):
            for k_frame in range(images["n_frames"]):
                shutil.copy(
                    src=images["folder"]+"/"+images["basename"]                 +"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                    dst=images["folder"]+"/"+images["basename"]+"_prenormalized"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])

        dwarp.compute_normalized_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            images_datatype=images["data_type"],
            verbose=verbose)
