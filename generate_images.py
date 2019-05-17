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

def generateImages(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        generate_image_gradient=False,
        verbose=0):

    mypy.my_print(verbose, "*** generateImages ***")

    assert ("n_integration" not in images),\
        "\"n_integration\" has been deprecated. Use \"resampling\" instead. Aborting."

    if ("resampling_factors" not in images):
        images["resampling_factors"] = [1]*images["n_dim"]
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

    if   (images["n_dim"] == 1):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling_factors"][0]-1, 0,                                               0, 0,                                               0])
    elif (images["n_dim"] == 2):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling_factors"][0]-1, 0, images["n_voxels"][1]*images["resampling_factors"][1]-1, 0,                                               0])
    elif (images["n_dim"] == 3):
        vtk_image.SetExtent([0, images["n_voxels"][0]*images["resampling_factors"][0]-1, 0, images["n_voxels"][1]*images["resampling_factors"][1]-1, 0, images["n_voxels"][2]*images["resampling_factors"][2]-1])
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."

    if   (images["n_dim"] == 1):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling_factors"][0],                                                           1.,                                                           1.])
    elif (images["n_dim"] == 2):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling_factors"][0], images["L"][1]/images["n_voxels"][1]/images["resampling_factors"][1],                                                           1.])
    elif (images["n_dim"] == 3):
        spacing = numpy.array([images["L"][0]/images["n_voxels"][0]/images["resampling_factors"][0], images["L"][1]/images["n_voxels"][1]/images["resampling_factors"][1], images["L"][2]/images["n_voxels"][2]/images["resampling_factors"][2]])
    vtk_image.SetSpacing(spacing)

    if   (images["n_dim"] == 1):
        origin = numpy.array([spacing[0]/2,           0.,           0.])
    elif (images["n_dim"] == 2):
        origin = numpy.array([spacing[0]/2, spacing[1]/2,           0.])
    elif (images["n_dim"] == 3):
        origin = numpy.array([spacing[0]/2, spacing[1]/2, spacing[2]/2])
    vtk_image.SetOrigin(origin)

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
        print "t = "+str(t)
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
    print "global_min = "+str(global_min)
    print "global_max = "+str(global_max)

    if (images["resampling_factors"] == [1]*images["n_dim"]):
        resampling = False
    else:
        resampling = True
        ddic.resample_images(
            images_folder=images["folder"],
            images_basename=images["basename"],
            resampling_factors=images["resampling_factors"],
            write_temp_images=1)

    if (images["data_type"] in ("float")):
        pass
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float", "uint8", "uint16", "uint32", "uint64", "ufloat")):
        ddic.normalize_images(
            images_folder=images["folder"],
            images_basename=images["basename"]+("_resampled"*(resampling)),
            images_datatype=images["data_type"])
