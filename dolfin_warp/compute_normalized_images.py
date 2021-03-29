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
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

import dolfin_warp as dwarp

################################################################################

def compute_normalized_images(
        images_folder,
        images_basename,
        images_datatype,
        images_ext="vti",
        suffix="",
        verbose=0):

    mypy.my_print(verbose, "*** compute_normalized_images ***")

    images_filenames = glob.glob(images_folder+"/"+images_basename+"_[0-9]*"+"."+images_ext)
    images_nframes = len(images_filenames)
    images_zfill = len(images_filenames[0].rsplit("_",1)[-1].split(".",1)[0])

    image_filename = images_folder+"/"+images_basename+"_"+str(0).zfill(images_zfill)+"."+images_ext
    image = myvtk.readImage(
        filename=image_filename,
        verbose=0)
    images_npoints = image.GetNumberOfPoints()

    if   (images_ext == "vtk"):
        reader = vtk.vtkImageReader()
        writer = vtk.vtkImageWriter()
    elif (images_ext == "vti"):
        reader = vtk.vtkXMLImageDataReader()
        writer = vtk.vtkXMLImageDataWriter()
    else:
        assert 0, "\"ext\" must be .vtk or .vti. Aborting."

    global_min = float("+Inf")
    global_max = float("-Inf")
    for k_frame in range(images_nframes):
        reader.SetFileName(images_folder+"/"+images_basename+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        reader.Update()

        image_scalars = reader.GetOutput().GetPointData().GetScalars()
        I = numpy.empty(image_scalars.GetNumberOfComponents())
        for k_point in range(images_npoints):
            image_scalars.GetTuple(k_point, I)

            global_min = min(global_min, I[0])
            global_max = max(global_max, I[0])
    mypy.my_print(verbose, "global_min = "+str(global_min))
    mypy.my_print(verbose, "global_max = "+str(global_max))

    shifter = vtk.vtkImageShiftScale()
    shifter.SetInputConnection(reader.GetOutputPort())
    shifter.SetShift(-global_min)
    if   (images_datatype in ("unsigned char", "uint8")):
        shifter.SetScale(float(2**8-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedChar()
    elif (images_datatype in ("unsigned short", "uint16")):
        shifter.SetScale(float(2**16-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedShort()
    elif (images_datatype in ("unsigned int", "uint32")):
        shifter.SetScale(float(2**32-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedInt()
    elif (images_datatype in ("unsigned long", "uint64")):
        shifter.SetScale(float(2**64-1)/(global_max-global_min))
        shifter.SetOutputScalarTypeToUnsignedLong()
    elif (images_datatype in ("unsigned float", "ufloat", "float")):
        shifter.SetScale(1./(global_max-global_min))
        shifter.SetOutputScalarTypeToFloat()

    writer.SetInputConnection(shifter.GetOutputPort())

    for k_frame in range(images_nframes):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        reader.SetFileName(images_folder+"/"+images_basename              +"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        writer.SetFileName(images_folder+"/"+images_basename+("_"+suffix)*(suffix!="")+"_"+str(k_frame).zfill(images_zfill)+"."+images_ext)
        writer.Write()
