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
from generate_images_Image import Image
from generate_images_Mapping import Mapping

################################################################################

def set_I_woGrad(
        image,
        X,
        I,
        vtk_image_scalars,
        k_point,
        G=None,
        Finv=None,
        vtk_gradient_vectors=None):

    image.I0(X, I)
    vtk_image_scalars.SetTuple(k_point, I)

def set_I_wGrad(
        image,
        X,
        I,
        vtk_image_scalars,
        k_point,
        G,
        Finv,
        vtk_gradient_vectors):

    image.I0_wGrad(X, I, G)
    vtk_image_scalars.SetTuple(k_point, I)
    G = numpy.dot(G, Finv)
    vtk_gradient_vectors.SetTuple(k_point, G)

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

    vtk_image = vtk.vtkImageData()

    delta = numpy.array(images["L"])/numpy.array(images["n_voxels"])

    images["n_voxels_upsampled"] = numpy.array(images["n_voxels"])*numpy.array(images["upsampling_factors"])
    delta_upsampled = numpy.array(images["L"])/numpy.array(images["n_voxels_upsampled"])

    if   (images["n_dim"] == 1):
        extent = [0, images["n_voxels_upsampled"][0]-1, 0,                                 0, 0,                                 0]
    elif (images["n_dim"] == 2):
        extent = [0, images["n_voxels_upsampled"][0]-1, 0, images["n_voxels_upsampled"][1]-1, 0,                                 0]
    elif (images["n_dim"] == 3):
        extent = [0, images["n_voxels_upsampled"][0]-1, 0, images["n_voxels_upsampled"][1]-1, 0, images["n_voxels_upsampled"][2]-1]
    vtk_image.SetExtent(extent)
    mypy.my_print(verbose, "extent = "+str(extent))

    if   (images["n_dim"] == 1):
        spacing = [delta_upsampled[0],                 1.,                 1.]
    elif (images["n_dim"] == 2):
        spacing = [delta_upsampled[0], delta_upsampled[1],                 1.]
    elif (images["n_dim"] == 3):
        spacing = [delta_upsampled[0], delta_upsampled[1], delta_upsampled[2]]
    vtk_image.SetSpacing(spacing)
    mypy.my_print(verbose, "spacing = "+str(spacing))

    if   (images["n_dim"] == 1):
        origin = [delta[0]/2,         0.,         0.]
    elif (images["n_dim"] == 2):
        origin = [delta[0]/2, delta[1]/2,         0.]
    elif (images["n_dim"] == 3):
        origin = [delta[0]/2, delta[1]/2, delta[2]/2]
    vtk_image.SetOrigin(origin)
    mypy.my_print(verbose, "origin = "+str(origin))

    n_points = vtk_image.GetNumberOfPoints()
    vtk_image_scalars = myvtk.createFloatArray(
        name="ImageScalars",
        n_components=1,
        n_tuples=n_points,
        verbose=verbose-1)
    vtk_image.GetPointData().SetScalars(vtk_image_scalars)

    if (generate_image_gradient):
        vtk_gradient = vtk.vtkImageData()
        vtk_gradient.DeepCopy(vtk_image)

        vtk_gradient_vectors = myvtk.createFloatArray(
            name="ImageScalarsGradient",
            n_components=3,
            n_tuples=n_points,
            verbose=verbose-1)
        vtk_gradient.GetPointData().SetScalars(vtk_gradient_vectors)
    else:
        vtk_gradient         = None
        vtk_gradient_vectors = None

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

    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mypy.my_print(verbose, "t = "+str(t))
        mapping.init_t(t)
        for k_point in xrange(n_points):
            vtk_image.GetPoint(k_point, x)
            #print "x0 = "+str(x)
            mapping.X(x, X, Finv)
            #print "X = "+str(X)
            set_I(image, X, I, vtk_image_scalars, k_point, G, Finv, vtk_gradient_vectors)
            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
        #print vtk_image
        myvtk.writeImage(
            image=vtk_image,
            filename=images["folder"]+"/"+images["basename"]+"_"+str(k_frame).zfill(images["zfill"])+"."+images["ext"],
            verbose=verbose-1)
        if (generate_image_gradient):
            #print vtk_gradient
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
        ddic.downsample_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            downsampling_factors=images["upsampling_factors"],
            keep_resolution=0,
            write_temp_images=0,
            verbose=verbose)

    if (images["data_type"] in ("float")):
        pass
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float", "uint8", "uint16", "uint32", "uint64", "ufloat")):
        ddic.normalize_images(
            images_folder=images["folder"],
            images_basename=images["basename"]+("_downsampled"*(downsampling)),
            images_datatype=images["data_type"],
            verbose=verbose)
