#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2016-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

from builtins import *

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
from .generate_images_Image   import Image
from .generate_images_Mapping import Mapping

################################################################################

def set_I_woGrad(
        image,
        X,
        I,
        image_upsampled_scalars,
        k_point,
        G=None,
        Finv=None,
        image_gradient_upsampled_vectors=None):

    image.I0(X, I)
    image_upsampled_scalars.SetTuple(k_point, I)

def set_I_wGrad(
        image,
        X,
        I,
        image_upsampled_scalars,
        k_point,
        G,
        Finv,
        image_gradient_upsampled_vectors):

    image.I0_wGrad(X, I, G)
    image_upsampled_scalars.SetTuple(k_point, I)
    G = numpy.dot(G, Finv)
    image_gradient_upsampled_vectors.SetTuple(k_point, G)

def generate_images(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        generate_image_gradient=False,
        verbose=0):

    mypy.my_print(verbose, "*** generate_images ***")

    assert ("n_integration" not in images),\
        "\"n_integration\" has been deprecated. Use \"upsampling\" instead. Aborting."

    if ("upsampling_factors" not in images):
        images["upsampling_factors"] = [1]*images["n_dim"]
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
        generate_image_gradient)
    mapping = Mapping(
        images,
        structure,
        deformation,
        evolution,
        generate_image_gradient)

    image_upsampled = vtk.vtkImageData()

    n_voxels_upsampled = list(numpy.multiply(images["n_voxels"], images["upsampling_factors"]))

    dimensions_upsampled = n_voxels_upsampled+[1]*(3-images["n_dim"])
    mypy.my_print(verbose, "dimensions_upsampled = "+str(dimensions_upsampled))
    image_upsampled.SetDimensions(dimensions_upsampled)

    delta = list(numpy.divide(images["L"], images["n_voxels"]))
    delta_upsampled = list(numpy.divide(delta, images["upsampling_factors"]))

    spacing_upsampled = delta_upsampled+[1.]*(3-images["n_dim"])
    mypy.my_print(verbose, "spacing_upsampled = "+str(spacing_upsampled))
    image_upsampled.SetSpacing(spacing_upsampled)

    origin_upsampled = list(numpy.divide(delta, 2))
    origin_upsampled = origin_upsampled+[0.]*(3-images["n_dim"])
    mypy.my_print(verbose, "origin_upsampled = "+str(origin_upsampled))
    image_upsampled.SetOrigin(origin_upsampled)

    n_points_upsampled = image_upsampled.GetNumberOfPoints()
    image_upsampled_scalars = myvtk.createFloatArray(
        name="ImageScalars",
        n_components=1,
        n_tuples=n_points_upsampled,
        verbose=verbose-1)
    image_upsampled.GetPointData().SetScalars(image_upsampled_scalars)

    if (generate_image_gradient):
        image_gradient_upsampled = vtk.vtkImageData()
        image_gradient_upsampled.DeepCopy(image_upsampled)

        image_gradient_upsampled_vectors = myvtk.createFloatArray(
            name="ImageScalarsGradient",
            n_components=3,
            n_tuples=n_points_upsampled,
            verbose=verbose-1)
        image_gradient_upsampled.GetPointData().SetScalars(image_gradient_upsampled_vectors)
    else:
        image_gradient_upsampled         = None
        image_gradient_upsampled_vectors = None

    x = numpy.empty(3)
    X = numpy.empty(3)
    I = numpy.empty(1)
    global_min = float("+Inf")
    global_max = float("-Inf")
    if (generate_image_gradient):
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
            image_upsampled.GetPoint(k_point, x)
            #print("x0 = "+str(x))
            mapping.X(x, X, Finv)
            #print("X = "+str(X))
            set_I(image, X, I, image_upsampled_scalars, k_point, G, Finv, image_gradient_upsampled_vectors)
            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
        #print(image_upsampled)
        myvtk.writeImage(
            image=image_upsampled,
            filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
            verbose=verbose-1)
        if (generate_image_gradient):
            #print(image_gradient_upsampled)
            myvtk.writeImage(
                image=image_gradient_upsampled,
                filename=images["folder"]+"/"+images["basename"]+"-grad"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                verbose=verbose-1)
    # mypy.my_print(verbose, "global_min = "+str(global_min))
    # mypy.my_print(verbose, "global_max = "+str(global_max))

    if (images["upsampling_factors"] == [1]*images["n_dim"]):
        downsampling = False
    else:
        downsampling = True

        ddic.downsample_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            downsampling_factors=images["upsampling_factors"],
            keep_resolution=0,
            write_temp_images=0,
            verbose=verbose)

        for k_frame in range(images["n_frames"]):
            os.rename(
                images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                images["folder"]+"/"+images["basename"]+"_upsampled"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])
            os.rename(
                images["folder"]+"/"+images["basename"]+"_downsampled"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])

    if (images["data_type"] in ("float")):
        normalizing = False
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float", "uint8", "uint16", "uint32", "uint64", "ufloat")):
        normalizing = True

        ddic.normalize_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            images_datatype=images["data_type"],
            verbose=verbose)

        for k_frame in range(images["n_frames"]):
            os.rename(
                src=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                dst=images["folder"]+"/"+images["basename"]+"_prenormalized"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])
            os.rename(
                src=images["folder"]+"/"+images["basename"]+"_normalized"+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
                dst=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"])
